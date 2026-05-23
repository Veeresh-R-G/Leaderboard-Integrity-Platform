"""
api/kafka_consumer.py
----------------------
Consumes activity events from Kafka, scores them,
stores results in PostgreSQL + Redis cache.

This is the real-time inference pipeline.
"""

import json
import os
import redis
import pandas as pd
from kafka import KafkaConsumer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from inference import AnomalyScorer

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
engine = create_engine(DB_URL)
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC_RAW", "activity.raw")


def fetch_gps_stream(activity_id: int) -> pd.DataFrame:
    """Pull GPS stream from TimescaleDB for scoring."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT time, speed_ms, hr_bpm, cadence_rpm, grade_pct
            FROM gps_streams
            WHERE activity_id = :activity_id
            ORDER BY time ASC
        """), {"activity_id": activity_id})
        df = pd.DataFrame(result.fetchall(),
                          columns=result.keys())
    return df


def store_score(activity_id: int, score_result: dict):
    """Persist anomaly score to PostgreSQL."""
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO anomaly_scores
                (activity_id, model_name, model_version,
                 score, is_flagged, flag_reason)
            VALUES
                (:activity_id, 'ts2vec', 'v1',
                 :score, :is_flagged, :reason)
            ON CONFLICT DO NOTHING
        """), {
            "activity_id": activity_id,
            "score":       score_result["score"],
            "is_flagged":  score_result["is_flagged"],
            "reason":      score_result["reason"],
        })
        conn.commit()


def cache_score(activity_id: int, score_result: dict):
    """Cache score in Redis for fast leaderboard lookup."""
    key = f"anomaly:score:{activity_id}"
    data = json.dumps({
        "score":      score_result["score"],
        "is_flagged": score_result["is_flagged"],
        "reason":     score_result["reason"],
    })
    redis_client.setex(key, 3600, data)   # TTL: 1 hour


def run_consumer():
    """Main consumer loop."""
    scorer = AnomalyScorer.get_instance(model_dir="models/saved")

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKER,
        group_id="anomaly-detector",
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode()),
        max_poll_records=10,
    )

    print(f"🎧 Listening on {TOPIC}...")
    processed = 0

    for message in consumer:
        event = message.value
        activity_id = event["activity_id"]

        try:
            # Pull GPS data from TimescaleDB
            gps_df = fetch_gps_stream(activity_id)

            if len(gps_df) < 30:
                print(f"  ⚠️  Activity {activity_id}: "
                      f"insufficient GPS data ({len(gps_df)} points)")
                continue

            # Score with TS2Vec
            result = scorer.score_activity(gps_df, activity_id)

            # Persist + cache
            store_score(activity_id, result)
            cache_score(activity_id, result)

            flag = "🚨 FLAGGED" if result["is_flagged"] else "✅ clean"
            processed += 1
            print(f"  [{processed}] Activity {activity_id}: "
                  f"score={result['score']:.4f}  {flag}  "
                  f"reason={result['reason']}")

        except Exception as e:
            print(f"  ❌ Error processing {activity_id}: {e}")
            continue


if __name__ == "__main__":
    run_consumer()
