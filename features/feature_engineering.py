"""
features/feature_engineering.py
---------------------------------
Converts raw GPS time-series into tabular features for ML models.

This is the most important file in the baseline pipeline.
The quality of features determines the ceiling of what
XGBoost / Logistic Regression can achieve.

Feature groups:
  1. Speed features        — distribution of speed over the activity
  2. Acceleration features — how quickly speed changes (cars accelerate differently)
  3. GPS quality features  — signal dropout, noise, point density
  4. Physiological features — HR vs speed relationship (key for car/ebike detection)
  5. Cadence features      — distinguishes running from cycling
  6. Statistical features  — higher-order moments of the speed distribution

Why these features?
  Strava's system uses "57 different factors like speed and acceleration".
  We engineer the most discriminative subset of those based on
  known anomaly characteristics from their engineering blog.

Reference: James @ Strava, March 2026
  "keeping-stravas-segment-leaderboards-fair-an-engineers-perspective"
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ── Thresholds (sports science + Strava engineering blog) ─────────
SPEED_CAR_THRESHOLD_MS    = 15.0   # > 15 m/s (54 km/h) = likely car
SPEED_EBIKE_THRESHOLD_MS  = 8.0    # > 8 m/s (29 km/h) sustained = likely ebike
SPEED_RUN_MAX_MS          = 6.5    # World record ~10.4 m/s, realistic max ~6.5
SPEED_RIDE_MAX_MS         = 22.0   # Realistic cycling max without motor
HR_MIN_EFFORT             = 100    # Below this at speed > 5 m/s = suspicious
CADENCE_RUN_MIN           = 140    # Below this for a "run" = suspicious
CADENCE_RIDE_MAX          = 120    # Above this for a "ride" = suspicious
MAX_HUMAN_ACCELERATION    = 3.0    # m/s² — sprinters peak ~3.5, but sustained < 2


class FeatureEngineer:
    """
    Computes per-activity features from GPS time-series.

    Input:  DataFrame with columns [activity_id, speed_ms, hr_bpm,
                                     cadence_rpm, grade_pct, timestamp]
    Output: One row per activity with engineered features
    """

    def compute_all(self, streams_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for all activities in the streams DataFrame.

        Args:
            streams_df: GPS streams with one row per GPS point

        Returns:
            features_df: One row per activity with all features
        """
        print("Computing features...")
        all_features = []

        grouped = streams_df.groupby("activity_id")
        total   = len(grouped)

        for i, (activity_id, group) in enumerate(grouped):
            if i % 100 == 0:
                print(f"  {i}/{total} activities processed...")

            group = group.sort_values("timestamp").reset_index(drop=True)
            features = self._compute_activity_features(activity_id, group)

            # Carry forward labels if present
            if "anomaly_type" in group.columns:
                features["anomaly_type"]  = group["anomaly_type"].iloc[0]
                features["anomaly_label"] = group["anomaly_label"].iloc[0]
            if "sport_type" in group.columns:
                features["sport_type"] = group["sport_type"].iloc[0]

            all_features.append(features)

        features_df = pd.DataFrame(all_features)
        print(f"\n✅ Features computed for {len(features_df)} activities")
        print(f"   Feature columns: {len(features_df.columns)}")
        return features_df

    def _compute_activity_features(
        self, activity_id: str, group: pd.DataFrame
    ) -> dict:
        """Compute all features for a single activity."""
        features = {"activity_id": activity_id}

        speed    = group["speed_ms"].values
        hr       = group["hr_bpm"].values
        cadence  = group["cadence_rpm"].values
        grade    = group["grade_pct"].values
        n_points = len(group)

        # ── 1. Speed features ─────────────────────────────────────
        features.update(self._speed_features(speed))

        # ── 2. Acceleration features ──────────────────────────────
        features.update(self._acceleration_features(speed))

        # ── 3. GPS quality features ───────────────────────────────
        features.update(self._gps_quality_features(group, speed))

        # ── 4. Physiological features ─────────────────────────────
        features.update(self._physiological_features(speed, hr))

        # ── 5. Cadence features ───────────────────────────────────
        features.update(self._cadence_features(cadence, speed))

        # ── 6. Grade / terrain features ───────────────────────────
        features.update(self._grade_features(grade, speed))

        # ── 7. Activity-level stats ───────────────────────────────
        features["duration_min"]    = n_points / 60.0
        features["total_distance_km"] = np.sum(speed) / 1000.0

        return features

    # ── Feature group methods ─────────────────────────────────────

    def _speed_features(self, speed: np.ndarray) -> dict:
        """
        Speed distribution features.

        Key insight: car GPS has high mean + high variance.
        E-bike has high mean + LOW variance (motor is consistent).
        Normal run has low mean + moderate variance.
        """
        n = len(speed)
        if n == 0:
            return {}

        return {
            # Central tendency
            "speed_mean_ms":        np.mean(speed),
            "speed_median_ms":      np.median(speed),

            # Spread
            "speed_std_ms":         np.std(speed),
            "speed_iqr_ms":         np.percentile(speed, 75) -
                                    np.percentile(speed, 25),

            # Extremes
            "speed_max_ms":         np.max(speed),
            "speed_p95_ms":         np.percentile(speed, 95),
            "speed_p99_ms":         np.percentile(speed, 99),

            # Threshold exceedance (key features for rule-based comparison)
            "pct_above_car_thresh":  np.mean(speed > SPEED_CAR_THRESHOLD_MS),
            "pct_above_ebike_thresh": np.mean(speed > SPEED_EBIKE_THRESHOLD_MS),
            "pct_above_run_max":     np.mean(speed > SPEED_RUN_MAX_MS),

            # Stops (traffic lights = car signal)
            "pct_stopped":          np.mean(speed < 0.5),
            "n_stop_events":        int(np.sum(
                np.diff((speed < 0.5).astype(int)) == 1)),

            # Shape of distribution
            "speed_skewness":       float(stats.skew(speed)),
            "speed_kurtosis":       float(stats.kurtosis(speed)),

            # Consistency — low std/mean = very consistent speed = ebike signal
            "speed_cv":             np.std(speed) / (np.mean(speed) + 1e-6),
        }

    def _acceleration_features(self, speed: np.ndarray) -> dict:
        """
        Acceleration and jerk features.

        Key insight: Cars have sharp acceleration/deceleration events.
        Human runners have smooth, gradual speed changes.
        Jerk (rate of acceleration change) is very high for cars at traffic lights.
        """
        if len(speed) < 3:
            return {}

        # First derivative: acceleration (m/s²)
        accel = np.diff(speed)         # change per second

        # Second derivative: jerk (m/s³) — rate of acceleration change
        jerk  = np.diff(accel)

        return {
            "accel_mean":           np.mean(np.abs(accel)),
            "accel_max":            np.max(np.abs(accel)),
            "accel_std":            np.std(accel),

            # Positive acceleration (speeding up)
            "accel_pos_mean":       np.mean(accel[accel > 0]) if
                                    np.any(accel > 0) else 0.0,

            # Negative acceleration (braking)
            "accel_neg_mean":       np.mean(np.abs(accel[accel < 0])) if
                                    np.any(accel < 0) else 0.0,

            # Superhuman acceleration events
            "n_superhuman_accel":   int(np.sum(
                np.abs(accel) > MAX_HUMAN_ACCELERATION)),
            "pct_superhuman_accel": np.mean(
                np.abs(accel) > MAX_HUMAN_ACCELERATION),

            # Jerk — key car signal (sharp stops at traffic lights)
            "jerk_mean":            np.mean(np.abs(jerk)),
            "jerk_max":             np.max(np.abs(jerk)),
            "jerk_p95":             np.percentile(np.abs(jerk), 95),

            # Sharp braking events (speed drops > 3 m/s in 1 second)
            "n_sharp_braking":      int(np.sum(accel < -3.0)),
        }

    def _gps_quality_features(
        self, group: pd.DataFrame, speed: np.ndarray
    ) -> dict:
        """
        GPS signal quality features.

        Key insight: GPS corruption produces impossible speed spikes
        and large time gaps between points.
        """
        n = len(group)
        if n < 2:
            return {}

        # Time gaps between consecutive GPS points
        try:
            timestamps = pd.to_datetime(group["timestamp"])
            gaps_s = timestamps.diff().dt.total_seconds().dropna().values
        except Exception:
            gaps_s = np.ones(n - 1)

        return {
            # Point density
            "n_gps_points":         n,
            "mean_gap_s":           np.mean(gaps_s),
            "max_gap_s":            np.max(gaps_s),
            "std_gap_s":            np.std(gaps_s),

            # Dropout events (gap > 10 seconds = signal lost)
            "n_dropout_events":     int(np.sum(gaps_s > 10)),
            "pct_dropout":          np.mean(gaps_s > 10),

            # Speed spike detection (GPS teleportation)
            # A spike is a value > 5 std from rolling median
            "n_speed_spikes":       self._count_speed_spikes(speed),

            # Positional noise — variance during near-stationary periods
            "positional_noise":     self._compute_positional_noise(
                group, speed),
        }

    def _physiological_features(
        self, speed: np.ndarray, hr: np.ndarray
    ) -> dict:
        """
        HR vs speed relationship features.

        Key insight: In real exercise, HR correlates with speed.
        In car GPS, HR is low (60-90) despite high speed.
        In e-bike, HR is moderate (90-120) despite high speed.

        This is the MOST discriminative feature for car/ebike detection.
        """
        if len(speed) < 10 or len(hr) < 10:
            return {}

        # Filter to moving periods only
        moving_mask = speed > 0.5
        if np.sum(moving_mask) < 10:
            return {}

        speed_moving = speed[moving_mask]
        hr_moving    = hr[moving_mask]

        # HR-speed correlation — should be strongly positive for real effort
        try:
            corr, pval = stats.pearsonr(speed_moving, hr_moving)
        except Exception:
            corr, pval = 0.0, 1.0

        # HR at high speed — car/ebike have low HR at high speed
        high_speed_mask = speed > 5.0
        hr_at_high_speed = np.mean(hr[high_speed_mask]) if \
            np.any(high_speed_mask) else np.mean(hr)

        # Impossible physiological combinations
        # Low HR + high speed = anomaly signal
        n_impossible = int(np.sum(
            (speed > 5.0) & (hr < HR_MIN_EFFORT)
        ))

        return {
            "hr_mean":              np.mean(hr),
            "hr_std":               np.std(hr),
            "hr_max":               np.max(hr),
            "hr_p95":               np.percentile(hr, 95),

            # The most important feature
            "hr_speed_correlation": float(corr),
            "hr_speed_corr_pval":   float(pval),

            "hr_at_high_speed":     float(hr_at_high_speed),
            "n_impossible_hr_speed": n_impossible,
            "pct_impossible_hr_speed": n_impossible / (len(speed) + 1e-6),

            # HR range — real exercise has wide range (warm up → hard effort)
            "hr_range":             np.max(hr) - np.min(hr),
        }

    def _cadence_features(
        self, cadence: np.ndarray, speed: np.ndarray
    ) -> dict:
        """
        Cadence features — key for wrong_sport_type detection.

        Running cadence: 155-185 spm
        Cycling cadence: 70-100 rpm
        Car/resting: 0-15

        Cadence alone separates running from cycling better than
        almost any other single feature.
        """
        if len(cadence) == 0:
            return {}

        moving_cadence = cadence[speed > 0.5] if np.any(speed > 0.5) else cadence

        return {
            "cadence_mean":          np.mean(cadence),
            "cadence_std":           np.std(cadence),
            "cadence_median":        np.median(cadence),
            "cadence_p25":           np.percentile(cadence, 25),
            "cadence_p75":           np.percentile(cadence, 75),

            # Zone classification
            "pct_cadence_run_zone":  np.mean(
                (cadence >= 140) & (cadence <= 210)),
            "pct_cadence_ride_zone": np.mean(
                (cadence >= 50) & (cadence < 140)),
            "pct_cadence_zero":      np.mean(cadence < 20),

            # Moving cadence (ignore stopped periods)
            "cadence_moving_mean":   np.mean(moving_cadence)
                                     if len(moving_cadence) > 0 else 0.0,

            # Cadence-speed ratio — different for running vs cycling
            "cadence_speed_ratio":   np.mean(cadence) /
                                     (np.mean(speed) + 1e-6),
        }

    def _grade_features(
        self, grade: np.ndarray, speed: np.ndarray
    ) -> dict:
        """
        Grade (road slope) vs speed relationship.

        Key insight: Humans slow down significantly on hills.
        Cars barely slow down on grades < 10%.
        E-bikes slow down less than normal cyclists.
        """
        if len(grade) < 2:
            return {}

        # Speed sensitivity to grade
        grade_abs = np.abs(grade)
        steep_mask = grade_abs > 5.0

        speed_on_flat  = np.mean(speed[grade_abs < 2]) if \
            np.any(grade_abs < 2) else np.mean(speed)
        speed_on_steep = np.mean(speed[steep_mask]) if \
            np.any(steep_mask) else speed_on_flat

        # Ratio: how much does speed drop on hills?
        # Real runners: ~0.6-0.8. Cars/ebike: ~0.9-1.0
        hill_speed_ratio = speed_on_steep / (speed_on_flat + 1e-6)

        return {
            "grade_mean":           np.mean(grade),
            "grade_std":            np.std(grade),
            "grade_max":            np.max(np.abs(grade)),
            "pct_steep":            np.mean(grade_abs > 5.0),
            "hill_speed_ratio":     float(hill_speed_ratio),
        }

    # ── Private helpers ───────────────────────────────────────────

    def _count_speed_spikes(self, speed: np.ndarray,
                             threshold_std: float = 5.0) -> int:
        """Count GPS teleportation events (impossible speed spikes)."""
        if len(speed) < 5:
            return 0
        rolling_med = pd.Series(speed).rolling(5, center=True).median().values
        residuals   = np.abs(speed - rolling_med)
        mad         = np.median(residuals) + 1e-6
        return int(np.sum(residuals > threshold_std * mad))

    def _compute_positional_noise(
        self, group: pd.DataFrame, speed: np.ndarray
    ) -> float:
        """Variance in GPS position during near-stationary periods."""
        if "lat" not in group.columns or "lon" not in group.columns:
            return 0.0
        stationary = speed < 0.3
        if np.sum(stationary) < 5:
            return 0.0
        lat_var = np.var(group["lat"].values[stationary])
        lon_var = np.var(group["lon"].values[stationary])
        # Convert to approximate metres
        return float((lat_var + lon_var) * 1e10)


def engineer_features(streams_path: str,
                       save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Main entry point — load streams CSV and compute features.

    Args:
        streams_path: path to gps_streams.csv
        save_path:    if provided, saves features CSV here

    Returns:
        features_df: one row per activity, ready for ML
    """
    print(f"Loading GPS streams from {streams_path}...")
    streams_df = pd.read_csv(streams_path)
    print(f"  Loaded {len(streams_df):,} GPS points across "
          f"{streams_df['activity_id'].nunique()} activities")

    engineer    = FeatureEngineer()
    features_df = engineer.compute_all(streams_df)

    if save_path:
        features_df.to_csv(save_path, index=False)
        print(f"  Saved features to {save_path}")

    return features_df


if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)

    features_df = engineer_features(
        streams_path="data/raw/gps_streams.csv",
        save_path="data/processed/features.csv",
    )

    print("\nFeature matrix shape:", features_df.shape)
    print("\nTop discriminating features by anomaly type:")
    print("(mean speed_mean_ms per class)")

    if "anomaly_label" in features_df.columns:
        summary = features_df.groupby("anomaly_label")[[
            "speed_mean_ms", "hr_mean", "cadence_mean",
            "hr_speed_correlation", "pct_above_car_thresh",
            "n_impossible_hr_speed",
        ]].mean().round(3)
        print(summary.to_string())