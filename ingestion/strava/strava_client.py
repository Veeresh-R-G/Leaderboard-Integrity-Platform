"""
ingestion/strava/strava_client.py
----------------------------------
Pulls your real Strava activity history and GPS streams.
Stores them in TimescaleDB for the inference pipeline.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
engine = create_engine(DB_URL)

ACCESS_TOKEN = os.getenv("STRAVA_ACCESS_TOKEN")
HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
BASE_URL = "https://www.strava.com/api/v3"


def get_athlete_activities(weeks_back: int = 12,
                           sport_type: str = "Run") -> list:
    """Pull activities from Strava API."""
    after_ts = int(
        (datetime.now() - timedelta(weeks=weeks_back)).timestamp()
    )
    activities = []
    page = 1

    while True:
        resp = requests.get(
            f"{BASE_URL}/athlete/activities",
            headers=HEADERS,
            params={"after": after_ts, "per_page": 100, "page": page}
        )
        if resp.status_code != 200:
            print(f"  API error: {resp.status_code}")
            break

        batch = resp.json()
        if not batch:
            break

        # Filter by sport type
        filtered = [a for a in batch
                    if a.get("type") == sport_type or
                    a.get("sport_type") == sport_type]
        activities.extend(filtered)
        page += 1
        time.sleep(0.5)  # rate limit: 100 req/15min

    print(f"  Fetched {len(activities)} {sport_type} activities")
    return activities


def get_activity_streams(activity_id: int) -> dict:
    """
    Pull GPS + sensor streams for one activity.
    Returns dict with time, latlng, altitude, velocity_smooth,
    heartrate, cadence, watts, grade_smooth.
    """
    resp = requests.get(
        f"{BASE_URL}/activities/{activity_id}/streams",
        headers=HEADERS,
        params={
            "keys": "time,latlng,altitude,velocity_smooth,"
                    "heartrate,cadence,watts,grade_smooth",
            "key_by_type": True,
        }
    )
    if resp.status_code != 200:
        return {}
    return resp.json()


def store_activity(activity: dict) -> int:
    """Insert activity metadata into PostgreSQL."""
    with engine.connect() as conn:
        # Upsert athlete
        conn.execute(text("""
            INSERT INTO athletes (strava_id, name)
            VALUES (:strava_id, :name)
            ON CONFLICT (strava_id) DO NOTHING
        """), {
            "strava_id": activity["athlete"]["id"],
            "name": activity.get("athlete", {}).get("firstname", "Unknown"),
        })

        # Upsert activity
        result = conn.execute(text("""
            INSERT INTO activities
                (strava_id, athlete_id, sport_type, start_time,
                 elapsed_time_s, distance_m, elevation_m,
                 device_type, source, anomaly_type)
            SELECT
                :strava_id,
                (SELECT athlete_id FROM athletes WHERE strava_id = :athlete_strava_id),
                :sport_type, :start_time, :elapsed_time_s,
                :distance_m, :elevation_m, :device_type,
                'strava', NULL
            ON CONFLICT (strava_id) DO UPDATE
                SET sport_type = EXCLUDED.sport_type
            RETURNING activity_id
        """), {
            "strava_id":      activity["id"],
            "athlete_strava_id": activity["athlete"]["id"],
            "sport_type":     activity.get("sport_type", "Run"),
            "start_time":     activity["start_date"],
            "elapsed_time_s": activity.get("elapsed_time", 0),
            "distance_m":     activity.get("distance", 0),
            "elevation_m":    activity.get("total_elevation_gain", 0),
            "device_type":    activity.get("device_name", "unknown"),
        })
        conn.commit()
        return result.fetchone()[0]


def store_gps_streams(activity_id: int,
                      streams: dict,
                      start_time: str) -> int:
    """Insert GPS streams into TimescaleDB hypertable."""
    if not streams or "time" not in streams:
        return 0

    start_dt = pd.to_datetime(start_time, utc=True)
    times = streams["time"]["data"]
    latlng = streams.get("latlng", {}).get("data", [])
    altitude = streams.get("altitude", {}).get("data", [None]*len(times))
    speed = streams.get("velocity_smooth", {}).get("data", [None]*len(times))
    hr = streams.get("heartrate", {}).get("data", [None]*len(times))
    cadence = streams.get("cadence", {}).get("data", [None]*len(times))
    grade = streams.get("grade_smooth", {}).get("data", [None]*len(times))

    records = []
    for i, t in enumerate(times):
        lat = latlng[i][0] if i < len(latlng) else None
        lon = latlng[i][1] if i < len(latlng) else None
        records.append({
            "time":        start_dt + timedelta(seconds=t),
            "activity_id": activity_id,
            "lat":         lat,
            "lon":         lon,
            "altitude_m":  altitude[i] if i < len(altitude) else None,
            "speed_ms":    speed[i] if i < len(speed) else None,
            "hr_bpm":      hr[i] if i < len(hr) else None,
            "cadence_rpm": cadence[i] if i < len(cadence) else None,
            "grade_pct":   grade[i] if i < len(grade) else None,
        })

    df = pd.DataFrame(records)
    df.to_sql("gps_streams", engine, if_exists="append",
              index=False, method="multi", chunksize=500)
    return len(records)


def ingest_strava_history(weeks_back: int = 12):
    """Full ingestion: pull Strava → store in TimescaleDB."""
    print(f"Ingesting Strava activity history ({weeks_back} weeks)...")
    activities = get_athlete_activities(weeks_back=weeks_back)

    total_points = 0
    for i, activity in enumerate(activities):
        print(f"  [{i+1}/{len(activities)}] "
              f"{activity['name'][:40]:<40} "
              f"{activity['distance']/1000:.1f}km")

        activity_id = store_activity(activity)
        streams = get_activity_streams(activity["id"])
        n_points = store_gps_streams(
            activity_id, streams, activity["start_date"])

        total_points += n_points
        time.sleep(0.5)   # rate limit

    print(f"\n✅ Ingestion complete")
    print(f"   Activities: {len(activities)}")
    print(f"   GPS points: {total_points:,}")


if __name__ == "__main__":
    ingest_strava_history(weeks_back=12)
