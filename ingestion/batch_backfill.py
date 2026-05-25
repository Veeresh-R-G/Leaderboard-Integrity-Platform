"""
ingestion/batch_backfill.py
-----------------------------
Scores ALL historical activities in TimescaleDB.

Mirrors what Strava does when they release a new anomaly
detection model and retroactively clean leaderboards.

Strava example:
  May 2025:  removed 4.45M anomalous run activities (backfill)
  Jan 2026:  removed 3.9M anomalous ride activities (backfill)
  Dec 2024:  removed 6.5M via rules update (backfill)

This script:
  1. Fetches all unscored activities from TimescaleDB
  2. Scores each with the TS2Vec model in batches
  3. Stores results in anomaly_scores table
  4. Prints a summary report

Usage:
  python ingestion/batch_backfill.py
  python ingestion/batch_backfill.py --model ts2vec --batch-size 50
  python ingestion/batch_backfill.py --rescore-all   # rescore even if already scored
"""

# Add parent directory to path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import time
import sys
import os
from api.inference import AnomalyScorer
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

load_dotenv()


DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
engine = create_engine(DB_URL)


def get_unscored_activities(model_name: str = "ts2vec",
                            rescore_all: bool = False,
                            limit: int = None) -> list:
    """
    Fetch activities that haven't been scored by this model yet.

    Args:
        model_name:  which model to check scores for
        rescore_all: if True, return ALL activities (re-score everything)
        limit:       max activities to return (None = all)
    """
    if rescore_all:
        query = """
            SELECT activity_id, sport_type, source, anomaly_type
            FROM activities
            ORDER BY created_at DESC
        """
    else:
        query = """
            SELECT a.activity_id, a.sport_type, a.source, a.anomaly_type
            FROM activities a
            LEFT JOIN anomaly_scores s
                ON a.activity_id = s.activity_id
                AND s.model_name = :model_name
            WHERE s.score_id IS NULL
            ORDER BY a.created_at DESC
        """

    if limit:
        query += f" LIMIT {limit}"

    with engine.connect() as conn:
        result = conn.execute(
            text(query),
            {"model_name": model_name} if not rescore_all else {}
        )
        return [dict(row._mapping) for row in result]


def fetch_gps_stream(activity_id: int) -> pd.DataFrame:
    """Pull GPS stream from TimescaleDB for one activity."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT time, speed_ms, hr_bpm, cadence_rpm, grade_pct
            FROM gps_streams
            WHERE activity_id = :activity_id
            ORDER BY time ASC
        """), {"activity_id": activity_id})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


def store_scores_batch(score_results: list,
                       model_name: str = "ts2vec"):
    """
    Bulk insert anomaly scores into PostgreSQL.
    Much faster than inserting one at a time.
    """
    if not score_results:
        return

    records = []
    for r in score_results:
        records.append({
            "activity_id": r["activity_id"],
            "model_name":  model_name,
            "model_version": "v1",
            "score":       r["score"],
            "is_flagged":  r["is_flagged"],
            "flag_reason": r["reason"],
        })

    df = pd.DataFrame(records)

    with engine.connect() as conn:
        # Use INSERT ... ON CONFLICT DO UPDATE for idempotency
        # Safe to run multiple times — won't create duplicates
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO anomaly_scores
                    (activity_id, model_name, model_version,
                     score, is_flagged, flag_reason)
                VALUES
                    (:activity_id, :model_name, :model_version,
                     :score, :is_flagged, :flag_reason)
                ON CONFLICT (activity_id, model_name)
                DO UPDATE SET
                    score        = EXCLUDED.score,
                    is_flagged   = EXCLUDED.is_flagged,
                    flag_reason  = EXCLUDED.flag_reason,
                    scored_at    = NOW()
            """), row.to_dict())
        conn.commit()


def score_activity_safe(scorer: AnomalyScorer,
                        activity: dict) -> dict:
    """
    Score one activity — catches errors so one bad activity
    doesn't stop the whole backfill.
    """
    activity_id = activity["activity_id"]

    try:
        gps_df = fetch_gps_stream(activity_id)

        if len(gps_df) < 30:
            return {
                "activity_id": activity_id,
                "score":       0.0,
                "is_flagged":  False,
                "reason":      "insufficient_gps_data",
                "skipped":     True,
            }

        result = scorer.score_activity(gps_df, str(activity_id))
        result["skipped"] = False
        return result

    except Exception as e:
        return {
            "activity_id": activity_id,
            "score":       0.0,
            "is_flagged":  False,
            "reason":      f"error: {str(e)[:100]}",
            "skipped":     True,
        }


def run_backfill(model_name:   str = "ts2vec",
                 batch_size:   int = 50,
                 rescore_all:  bool = False,
                 max_workers:  int = 4,
                 limit:        int = None):
    """
    Main backfill function.

    Args:
        model_name:  model to use for scoring
        batch_size:  activities per batch (controls memory usage)
        rescore_all: re-score activities that already have scores
        max_workers: parallel threads for GPS stream fetching
        limit:       cap total activities (useful for testing)
    """
    start_time = datetime.now()

    print("="*60)
    print("  BATCH BACKFILL — Leaderboard Integrity System")
    print("="*60)
    print(f"  Model:       {model_name}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Rescore all: {rescore_all}")
    print(f"  Workers:     {max_workers}")
    print(f"  Started:     {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Load model ────────────────────────────────────────────────
    print("\nLoading model...")
    scorer = AnomalyScorer.get_instance()

    # ── Get activities to score ───────────────────────────────────
    print("\nFetching unscored activities...")
    activities = get_unscored_activities(
        model_name=model_name,
        rescore_all=rescore_all,
        limit=limit,
    )

    if not activities:
        print("✅ Nothing to score — all activities already have scores")
        return

    print(f"  Found {len(activities):,} activities to score")

    # ── Score in batches ──────────────────────────────────────────
    total = len(activities)
    scored = 0
    flagged = 0
    skipped = 0
    errors = 0
    batch_results = []

    # Counters per anomaly type (for synthetic data with known labels)
    type_counts = {}
    type_flagged = {}

    print(f"\nScoring {total:,} activities in batches of {batch_size}...")
    print("─"*60)

    with tqdm(total=total, unit="activity") as pbar:
        for batch_start in range(0, total, batch_size):
            batch = activities[batch_start: batch_start + batch_size]

            # Score batch in parallel (I/O bound — fetching from DB)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(score_activity_safe, scorer, act): act
                    for act in batch
                }
                for future in as_completed(futures):
                    result = future.result()
                    activity = futures[future]

                    # Track statistics
                    if result.get("skipped"):
                        skipped += 1
                    else:
                        scored += 1
                        if result["is_flagged"]:
                            flagged += 1

                        # Track per anomaly type (if label known)
                        atype = activity.get("anomaly_type", "unknown")
                        if atype is not None:
                            atype = str(atype)
                            type_counts[atype] = type_counts.get(atype, 0) + 1
                            if result["is_flagged"]:
                                type_flagged[atype] = type_flagged.get(
                                    atype, 0) + 1

                    batch_results.append(result)
                    pbar.update(1)

            # Store completed batch to DB
            valid_results = [r for r in batch_results
                             if not r.get("skipped")]
            store_scores_batch(valid_results, model_name)
            batch_results = []   # clear for next batch

    # ── Summary report ────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    throughput = scored / max(elapsed, 1)

    print("\n" + "="*60)
    print("  BACKFILL COMPLETE")
    print("="*60)
    print(f"  Total activities:  {total:,}")
    print(f"  Scored:            {scored:,}")
    print(f"  Skipped:           {skipped:,}  (insufficient GPS data)")
    print(f"  Flagged:           {flagged:,}  ({flagged/max(scored, 1):.1%})")
    print(f"  Elapsed:           {elapsed:.1f}s")
    print(f"  Throughput:        {throughput:.1f} activities/second")

    # Per-type breakdown (for synthetic data)
    if type_counts:
        LABEL_NAMES = {
            "0": "normal", "1": "car_gps", "2": "ebike",
            "3": "gps_corruption", "4": "wrong_sport_type",
            "5": "partial_anomaly",
        }
        print(f"\n  Detection rate per type:")
        print(f"  {'Type':>22}  {'Flagged':>8}  {'Total':>7}  {'Rate':>7}")
        print(f"  {'─'*48}")
        for atype, count in sorted(type_counts.items()):
            label = LABEL_NAMES.get(atype, atype)
            n_flagged = type_flagged.get(atype, 0)
            rate = n_flagged / max(count, 1)
            flag_icon = "✅" if rate > 0.8 else "⚠️ " if rate > 0.5 else "❌"
            print(f"  {label:>22}  {n_flagged:>8}  {count:>7}  "
                  f"{rate:>6.1%}  {flag_icon}")

    print(f"\n  Scores stored in: anomaly_scores table")
    print(f"  Query results:    SELECT * FROM anomaly_scores "
          f"WHERE model_name = '{model_name}';")

    return {
        "total":      total,
        "scored":     scored,
        "flagged":    flagged,
        "skipped":    skipped,
        "elapsed_s":  elapsed,
        "throughput": throughput,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch backfill anomaly scores for all activities"
    )
    parser.add_argument("--model",       default="ts2vec",
                        help="Model name (ts2vec or tnc)")
    parser.add_argument("--batch-size",  type=int, default=50,
                        help="Activities per batch")
    parser.add_argument("--rescore-all", action="store_true",
                        help="Re-score activities that already have scores")
    parser.add_argument("--workers",     type=int, default=4,
                        help="Parallel threads for DB fetching")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Max activities to score (for testing)")
    args = parser.parse_args()

    run_backfill(
        model_name=args.model,
        batch_size=args.batch_size,
        rescore_all=args.rescore_all,
        max_workers=args.workers,
        limit=args.limit,
    )
