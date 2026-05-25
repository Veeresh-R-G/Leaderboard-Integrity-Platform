"""
api/main.py
------------
FastAPI inference service.
Exposes anomaly scores via REST API.
"""

import json
import os
import redis
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from api.inference import AnomalyScorer

load_dotenv()

app = FastAPI(
    title="Strava Leaderboard Integrity API",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
engine = create_engine(DB_URL)
redis_client = redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"))


# ── Request / Response models ─────────────────────────────────────

class GPSPoint(BaseModel):
    speed_ms:    float
    hr_bpm:      Optional[float] = 0.0
    cadence_rpm: Optional[float] = 0.0
    grade_pct:   Optional[float] = 0.0


class ScoreRequest(BaseModel):
    activity_id: Optional[str] = None
    gps_points:  List[GPSPoint]


class ScoreResponse(BaseModel):
    activity_id:   Optional[str]
    score:         float
    is_flagged:    bool
    reason:        str
    n_windows:     int
    confidence:    str


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status":  "ok",
        "service": "Strava Leaderboard Integrity",
        "version": "1.0.0",
    }


@app.post("/score", response_model=ScoreResponse)
def score_activity(request: ScoreRequest):
    """
    Score a GPS activity for anomalies.
    Accepts raw GPS points, returns anomaly score + reason.
    """
    scorer = AnomalyScorer.get_instance()

    # Check Redis cache first
    if request.activity_id:
        cached = redis_client.get(
            f"anomaly:score:{request.activity_id}")
        if cached:
            data = json.loads(cached)
            return ScoreResponse(
                activity_id=request.activity_id,
                score=data["score"],
                is_flagged=data["is_flagged"],
                reason=data["reason"],
                n_windows=0,
                confidence="cached",
            )

    # Convert to DataFrame
    gps_df = pd.DataFrame([p.dict() for p in request.gps_points])

    if len(gps_df) < 30:
        raise HTTPException(
            status_code=400,
            detail="Minimum 30 GPS points required"
        )

    result = scorer.score_activity(gps_df, request.activity_id)

    # Cache result
    if request.activity_id:
        redis_client.setex(
            f"anomaly:score:{request.activity_id}",
            3600,
            json.dumps({
                "score":      result["score"],
                "is_flagged": result["is_flagged"],
                "reason":     result["reason"],
            })
        )

    confidence = ("high" if abs(result["score"] - 0.5) > 0.3
                  else "medium" if abs(result["score"] - 0.5) > 0.1
                  else "low")

    return ScoreResponse(
        activity_id=request.activity_id,
        score=result["score"],
        is_flagged=result["is_flagged"],
        reason=result["reason"],
        n_windows=result["n_windows"],
        confidence=confidence,
    )


@app.get("/activity/{activity_id}/score")
def get_activity_score(activity_id: int):
    """Get cached anomaly score for a stored activity."""
    # Check Redis
    cached = redis_client.get(f"anomaly:score:{activity_id}")
    if cached:
        return {"source": "cache", **json.loads(cached)}

    # Check PostgreSQL
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT score, is_flagged, flag_reason, scored_at
            FROM anomaly_scores
            WHERE activity_id = :aid
            ORDER BY scored_at DESC LIMIT 1
        """), {"aid": activity_id})
        row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404,
                            detail="Activity not scored yet")

    return {
        "source":     "database",
        "score":      row.score,
        "is_flagged": row.is_flagged,
        "reason":     row.flag_reason,
        "scored_at":  str(row.scored_at),
    }


@app.get("/stats")
def get_stats():
    """Pipeline health + model performance stats."""
    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT
                COUNT(*)                              AS total_scored,
                SUM(CASE WHEN is_flagged THEN 1 END) AS total_flagged,
                AVG(score)                            AS avg_score,
                MAX(scored_at)                        AS last_scored
            FROM anomaly_scores
            WHERE model_name = 'ts2vec'
        """)).fetchone()

    return {
        "total_scored":  stats.total_scored,
        "total_flagged": stats.total_flagged,
        "flag_rate":     round(
            (stats.total_flagged or 0) /
            max(stats.total_scored or 1, 1) * 100, 2),
        "avg_score":     round(float(stats.avg_score or 0), 4),
        "last_scored":   str(stats.last_scored),
        "model":         "TS2Vec (self-supervised contrastive)",
    }
