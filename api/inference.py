"""
api/inference.py
-----------------
Loads saved models and scores GPS activities.

Used by:
  - Kafka consumer (real-time pipeline)
  - FastAPI endpoints (on-demand scoring)
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

INPUT_FEATURES = ["speed_ms", "hr_bpm", "cadence_rpm", "grade_pct"]
SEQ_LEN = 256
STRIDE = 128
N_FEATURES = len(INPUT_FEATURES)


# ── Encoder architecture (must match training) ────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class GPSEncoder(nn.Module):
    def __init__(self, n_features=N_FEATURES, d_model=128,
                 n_heads=4, n_layers=2, embed_dim=64, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, d_model//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model//2), nn.GELU(),
            nn.Conv1d(d_model//2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, embed_dim),
        )

    def forward(self, x):
        B, T, F = x.shape
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        h = self.pos_enc(h)
        h = self.transformer(h)
        z_temporal = self.projector(h)
        z_pooled = z_temporal.mean(dim=1)
        return {"temporal": z_temporal, "pooled": z_pooled}


class AnomalyScorer:
    """
    Loads saved model artifacts and scores GPS activities.
    Singleton pattern — load once, score many.
    """

    _instance = None

    @classmethod
    def get_instance(cls, model_dir: str = "models/saved"):
        if cls._instance is None:
            cls._instance = cls(model_dir)
        return cls._instance

    def __init__(self, model_dir: str = "models/saved"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        with open(f"{model_dir}/config.pkl", "rb") as f:
            self.config = pickle.load(f)

        # Load scaler
        with open(f"{model_dir}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        # Load TS2Vec encoder
        self.encoder = GPSEncoder(
            n_features=self.config["n_features"],
            d_model=self.config["d_model"],
            n_heads=self.config["n_heads"],
            n_layers=self.config["n_layers"],
            embed_dim=self.config["embed_dim"],
        ).to(self.device)
        self.encoder.load_state_dict(torch.load(
            f"{model_dir}/ts2vec_encoder.pt",
            map_location=self.device,
        ))
        self.encoder.eval()

        # Load normal centroid
        self.normal_centroid = np.load(
            f"{model_dir}/normal_centroid_ts2vec.npy")
        self.threshold = self.config["anomaly_threshold"]

        print(f"✅ AnomalyScorer loaded")
        print(f"   Device:    {self.device}")
        print(f"   Threshold: {self.threshold:.4f}")

    def score_activity(self,
                       gps_df,
                       activity_id: Optional[str] = None) -> dict:
        """
        Score a single GPS activity.

        Args:
            gps_df: DataFrame with columns matching INPUT_FEATURES
            activity_id: optional ID for logging

        Returns:
            dict with score, is_flagged, reason, window_scores
        """
        import pandas as pd

        # Preprocess
        X = gps_df[INPUT_FEATURES].fillna(0).replace(
            [float("inf"), float("-inf")], 0).values
        X = self.scaler.transform(X)

        # Extract windows
        seq_len = self.config["seq_len"]
        stride = self.config["stride"]
        windows = []

        if len(X) < seq_len:
            pad = np.zeros((seq_len - len(X), X.shape[1]))
            X = np.vstack([X, pad])

        for start in range(0, len(X) - seq_len + 1, stride):
            windows.append(X[start: start + seq_len])

        if not windows:
            return {"score": 0.0, "is_flagged": False,
                    "reason": "insufficient_data", "window_scores": []}

        windows_tensor = torch.FloatTensor(
            np.array(windows)).to(self.device)

        # Get embeddings
        with torch.no_grad():
            out = self.encoder(windows_tensor)
            embs = out["temporal"].mean(dim=1).cpu().numpy()

        # Compute distances from normal centroid
        distances = np.linalg.norm(
            embs - self.normal_centroid, axis=1)

        # Normalise (approximate — in production use training distribution)
        score = float(np.max(distances))
        is_flagged = score >= self.threshold

        # Generate reason
        reason = self._generate_reason(gps_df, score)

        return {
            "activity_id":   activity_id,
            "score":         round(score, 4),
            "is_flagged":    is_flagged,
            "reason":        reason,
            "n_windows":     len(windows),
            "window_scores": distances.tolist(),
            "max_window":    int(np.argmax(distances)),
        }

    def _generate_reason(self, gps_df, score: float) -> str:
        """Human-readable reason for flagging."""
        speed = gps_df["speed_ms"].values
        hr = gps_df["hr_bpm"].values   \
            if "hr_bpm" in gps_df.columns else np.zeros(len(speed))

        reasons = []
        if np.mean(speed > 15) > 0.10:
            reasons.append("sustained_high_speed")
        if np.mean((speed > 5) & (hr < 100)) > 0.15:
            reasons.append("low_hr_at_high_speed")
        if np.max(speed) > 25:
            reasons.append("impossible_speed_spike")
        if "cadence_rpm" in gps_df.columns:
            cad = gps_df["cadence_rpm"].values
            if np.mean((cad > 50) & (cad < 140)) > 0.6:
                reasons.append("cycling_cadence_on_run")

        return ", ".join(reasons) if reasons else "embedding_distance"
