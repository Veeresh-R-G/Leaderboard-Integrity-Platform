'''
This file is used to re configure the config to set the threshold to test the 
Model on the strava data

Initially we saved the threshold which was scaled between [0,1]
So min = 0(closest to centroid), max = 1(most probably anomalous)

But on my strava data, the scores were ranging from [15.7, 46.6]
'''

import os
import pickle
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
engine = create_engine(DB_URL)

# ── Get real Strava scores ────────────────────────────────────────
with engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT s.score
        FROM anomaly_scores s
        JOIN activities a USING (activity_id)
        WHERE s.model_name = 'ts2vec'
        AND a.source = 'strava'
        ORDER BY s.score
    """)).fetchall()

scores = [r.score for r in rows]
print(f"Real Strava run scores ({len(scores)} activities):")
print(f"  Min:    {min(scores):.2f}")
print(f"  Max:    {max(scores):.2f}")
print(f"  Mean:   {np.mean(scores):.2f}")
print(f"  Median: {np.median(scores):.2f}")
print(f"  P90:    {np.percentile(scores, 90):.2f}")
print(f"  P95:    {np.percentile(scores, 95):.2f}")
print(f"  P99:    {np.percentile(scores, 99):.2f}")
print()

# ── Set new threshold ─────────────────────────────────────────────
new_threshold = float(np.percentile(scores, 95))
n_flagged = sum(s > new_threshold for s in scores)
n_clean = sum(s <= new_threshold for s in scores)

print(f"New threshold (95th percentile): {new_threshold:.2f}")
print(f"  Flagged: {n_flagged}")
print(f"  Clean:   {n_clean}")
print()

# ── Update config.pkl ─────────────────────────────────────────────
with open("models/saved/config.pkl", "rb") as f:
    config = pickle.load(f)

old_threshold = config.get("anomaly_threshold")
config["anomaly_threshold"] = new_threshold
config["threshold_source"] = "calibrated_on_real_strava_data"
config["n_calibration_activities"] = len(scores)

with open("models/saved/config.pkl", "wb") as f:
    pickle.dump(config, f)

print(f"config.pkl updated:")
print(f"  Old threshold: {old_threshold:.4f}")
print(f"  New threshold: {new_threshold:.2f}")
print()

# ── Re-flag activities in database ────────────────────────────────
with engine.connect() as conn:
    conn.execute(text("""
        UPDATE anomaly_scores
        SET is_flagged = (score >= :threshold)
        WHERE model_name = 'ts2vec'
        AND activity_id IN (
            SELECT activity_id FROM activities
            WHERE source = 'strava'
        )
    """), {"threshold": new_threshold})
    conn.commit()

print("Database re-flagged with new threshold")
print()

# ── Show final results ────────────────────────────────────────────
with engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT
            a.activity_id,
            ROUND((a.distance_m / 1000)::numeric, 1) AS km,
            ROUND(s.score::numeric, 2)                AS score,
            s.is_flagged,
            s.flag_reason
        FROM anomaly_scores s
        JOIN activities a USING (activity_id)
        WHERE s.model_name = 'ts2vec'
        ORDER BY s.score DESC
    """)).fetchall()

print(f"{'Activity':>12}  {'KM':>6}  {'Score':>7}  {'Status':>8}  Reason")
print("-" * 70)
for r in rows:
    status = "FLAGGED" if r.is_flagged else "clean"
    reason = r.flag_reason or ""
    print(f"{str(r.activity_id):>12}  "
          f"{str(r.km):>6}  "
          f"{r.score:>7.2f}  "
          f"{status:>8}  "
          f"{reason}")
