"""
ingestion/kafka_producer.py
-----------------------------
Simulates activity upload events flowing into Kafka.

In production Strava:
  Every activity upload → event on Kafka topic
  ~85,000 events/minute at peak

We simulate this by reading from TimescaleDB
and publishing to Kafka with configurable rate.
"""

import json
import time
import os
from datetime import datetime
from kafka import KafkaProducer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
engine = create_engine(DB_URL)
BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC_RAW", "activity.raw")


def get_activities_to_score():
    """Get activities that haven't been scored yet."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT
                a.activity_id,
                a.strava_id,
                a.sport_type,
                a.start_time,
                a.distance_m,
                a.elapsed_time_s,
                a.source
            FROM activities a
            LEFT JOIN anomaly_scores s
                ON a.activity_id = s.activity_id
                AND s.model_name = 'ts2vec'
            WHERE s.score_id IS NULL
            ORDER BY a.created_at DESC
            LIMIT 100
        """))
        return [dict(row._mapping) for row in result]


def produce_activity_events(
    rate_per_second: float = 5.0,
    max_events: int = None,
):
    """
    Publish activity events to Kafka topic.

    Args:
        rate_per_second: events to publish per second
        max_events:      stop after this many (None = all)
    """
    producer = KafkaProducer(
        bootstrap_servers=BROKER,
        value_serializer=lambda v: json.dumps(v, default=str).encode(),
        acks="all",         # wait for all replicas
        retries=3,
        batch_size=16384,
        linger_ms=10,
    )

    activities = get_activities_to_score()
    if max_events:
        activities = activities[:max_events]

    print(f"Publishing {len(activities)} activity events to {TOPIC}")
    print(f"Rate: {rate_per_second} events/second")

    published = 0
    for activity in activities:
        event = {
            "event_type":  "activity.uploaded",
            "activity_id": activity["activity_id"],
            "strava_id":   activity["strava_id"],
            "sport_type":  activity["sport_type"],
            "distance_m":  activity["distance_m"],
            "source":      activity["source"],
            "timestamp":   datetime.utcnow().isoformat(),
        }

        producer.send(
            TOPIC,
            key=str(activity["activity_id"]).encode(),
            value=event,
        )
        published += 1

        if published % 10 == 0:
            print(f"  Published {published}/{len(activities)}")

        time.sleep(1.0 / rate_per_second)

    producer.flush()
    producer.close()
    print(f"\n✅ Published {published} events to {TOPIC}")


if __name__ == "__main__":
    produce_activity_events(rate_per_second=5.0)
