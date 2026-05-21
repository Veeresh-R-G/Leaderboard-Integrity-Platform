"""
ingestion/synthetic_generator.py
----------------------------------
Generates synthetic GPS activities for training baseline models.

Why synthetic data?
Real labelled anomaly data from Strava doesn't exist publicly.
We generate realistic normal + anomalous activities based on
known characteristics of each anomaly type.

Activity types generated:
  0: Normal run        — realistic human running pattern
  1: Car GPS           — high speed, traffic stops, wrong acceleration
  2: E-bike            — sustained 30-45 km/h, low HR despite speed
  3: GPS corruption    — teleportation jumps, signal dropout
  4: Wrong sport type  — cycling GPS labelled as run
  5: Partial anomaly   — real run + car GPS splice (hardest case)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import uuid


# ── Anomaly type constants ────────────────────────────────────────
NORMAL = 0
CAR_GPS = 1
EBIKE = 2
GPS_CORRUPT = 3
WRONG_SPORT = 4
PARTIAL = 5

ANOMALY_LABELS = {
    NORMAL:      "normal",
    CAR_GPS:     "car_gps",
    EBIKE:       "ebike",
    GPS_CORRUPT: "gps_corruption",
    WRONG_SPORT: "wrong_sport_type",
    PARTIAL:     "partial_anomaly",
}


@dataclass
class GPSPoint:
    """Single GPS data point at 1Hz resolution."""
    timestamp:   datetime
    lat:         float
    lon:         float
    altitude_m:  float
    speed_ms:    float      # speed in m/s
    hr_bpm:      int        # heart rate
    cadence_rpm: int        # steps/min (running) or rpm (cycling)
    grade_pct:   float      # road grade %


@dataclass
class SyntheticActivity:
    """A complete synthetic activity with metadata and GPS stream."""
    activity_id:  str
    anomaly_type: int
    anomaly_label: str
    sport_type:   str        # run / ride
    start_time:   datetime
    points:       List[GPSPoint] = field(default_factory=list)

    @property
    def duration_seconds(self) -> int:
        return len(self.points)

    @property
    def distance_m(self) -> float:
        if len(self.points) < 2:
            return 0.0
        # Haversine approximation — good enough for short segments
        total = 0.0
        for i in range(1, len(self.points)):
            total += _haversine(
                self.points[i-1].lat, self.points[i-1].lon,
                self.points[i].lat,   self.points[i].lon,
            )
        return total

    def to_dataframe(self) -> pd.DataFrame:
        """Convert GPS points to DataFrame for feature engineering."""
        records = []
        for p in self.points:
            records.append({
                "activity_id":  self.activity_id,
                "anomaly_type": self.anomaly_type,
                "anomaly_label": self.anomaly_label,
                "sport_type":   self.sport_type,
                "timestamp":    p.timestamp,
                "lat":          p.lat,
                "lon":          p.lon,
                "altitude_m":   p.altitude_m,
                "speed_ms":     p.speed_ms,
                "hr_bpm":       p.hr_bpm,
                "cadence_rpm":  p.cadence_rpm,
                "grade_pct":    p.grade_pct,
            })
        return pd.DataFrame(records)


def _haversine(lat1: float, lon1: float,
               lat2: float, lon2: float) -> float:
    """Distance in metres between two GPS coordinates."""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


class SyntheticGPSGenerator:
    """
    Generates realistic synthetic GPS activities.

    Each generator method produces one activity of a given type.
    Call generate_dataset() to produce a full labelled dataset.

    Starting coordinates: Bengaluru, India (12.97°N, 77.59°E)
    Activities simulate movement from this point.
    """

    # Starting point — Bengaluru
    BASE_LAT = 12.9716
    BASE_LON = 77.5946

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    # ── Public API ────────────────────────────────────────────────

    def generate_dataset(self,
                         n_per_class: int = 200,
                         save_path: Optional[str] = None
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate a full labelled dataset.

        Args:
            n_per_class: number of activities per anomaly type
            save_path:   if provided, saves CSVs to this path

        Returns:
            (activities_df, streams_df) — metadata and GPS streams
        """
        all_activities = []
        all_streams = []

        generators = {
            NORMAL:      self.generate_normal_run,
            CAR_GPS:     self.generate_car_gps,
            EBIKE:       self.generate_ebike,
            GPS_CORRUPT: self.generate_gps_corruption,
            WRONG_SPORT: self.generate_wrong_sport,
            PARTIAL:     self.generate_partial_anomaly,
        }

        for anomaly_type, gen_fn in generators.items():
            label = ANOMALY_LABELS[anomaly_type]
            print(f"  Generating {n_per_class} × {label}...")
            for i in range(n_per_class):
                activity = gen_fn()
                stream_df = activity.to_dataframe()
                all_streams.append(stream_df)
                all_activities.append({
                    "activity_id":   activity.activity_id,
                    "anomaly_type":  activity.anomaly_type,
                    "anomaly_label": activity.anomaly_label,
                    "sport_type":    activity.sport_type,
                    "start_time":    activity.start_time,
                    "duration_s":    activity.duration_seconds,
                    "distance_m":    activity.distance_m,
                    "n_points":      len(activity.points),
                })

        activities_df = pd.DataFrame(all_activities)
        streams_df = pd.concat(all_streams, ignore_index=True)

        if save_path:
            import os
            os.makedirs(save_path, exist_ok=True)
            activities_df.to_csv(f"{save_path}/activities.csv", index=False)
            streams_df.to_csv(f"{save_path}/gps_streams.csv", index=False)
            print(f"  Saved to {save_path}/")

        print(f"\n✅ Dataset generated:")
        print(f"   Activities: {len(activities_df)}")
        print(f"   GPS points: {len(streams_df)}")
        print(f"   Class distribution:")
        for label, count in activities_df["anomaly_label"].value_counts().items():
            print(f"     {label:25s}: {count}")

        return activities_df, streams_df

    # ── Generator methods ─────────────────────────────────────────

    def generate_normal_run(self) -> SyntheticActivity:
        """
        Normal human running activity.

        Characteristics:
        - Speed: 2.5 - 4.5 m/s (9 - 16 km/h)
        - HR: 130-175 bpm, correlated with speed
        - Cadence: 155-185 spm
        - Natural speed variation (hills, fatigue)
        - Duration: 25-90 minutes
        """
        duration_s = int(self.rng.uniform(25 * 60, 90 * 60))
        base_speed = self.rng.uniform(2.8, 4.2)  # m/s

        points = []
        lat, lon = self._random_start()
        alt = self.rng.uniform(800, 950)     # Bengaluru elevation
        t = self._random_start_time()

        for i in range(duration_s):
            # Natural speed variation — runners slow on hills, vary pace
            fatigue_factor = 1.0 - (i / duration_s) * 0.08  # ~8% slowdown
            hill_factor = 1.0 + 0.15 * np.sin(i / 180)   # hill undulation
            noise = self.rng.normal(0, 0.15)
            speed = max(1.5, base_speed * fatigue_factor * hill_factor + noise)

            # HR correlated with speed — key signal for authenticity
            hr = int(self.rng.normal(130 + speed * 12, 4))
            hr = np.clip(hr, 110, 195)

            cadence = int(self.rng.normal(168, 6))

            # Grade affects speed
            grade = self.rng.normal(0, 3.0)
            speed *= max(0.6, 1.0 - grade * 0.03)

            # Move position
            bearing = self.rng.normal(90, 20)   # mostly eastward
            lat, lon = self._move(lat, lon, speed, bearing)
            alt += self.rng.normal(0, 0.5)

            points.append(GPSPoint(
                timestamp=t + timedelta(seconds=i),
                lat=lat, lon=lon, altitude_m=alt,
                speed_ms=speed, hr_bpm=hr,
                cadence_rpm=cadence, grade_pct=grade,
            ))

        return SyntheticActivity(
            activity_id=str(uuid.uuid4())[:8],
            anomaly_type=NORMAL,
            anomaly_label=ANOMALY_LABELS[NORMAL],
            sport_type="run",
            start_time=t,
            points=points,
        )

    def generate_car_gps(self) -> SyntheticActivity:
        """
        Car recorded as a run/ride.

        Characteristics:
        - Speed: 8-30 m/s (30-108 km/h) with traffic patterns
        - Sharp acceleration / deceleration at traffic lights
        - Speed = 0 for 30-120 second intervals (red lights)
        - HR implausibly low for stated speed
        - High jerk (rate of acceleration change)
        """
        duration_s = int(self.rng.uniform(10 * 60, 40 * 60))

        points = []
        lat, lon = self._random_start()
        alt = self.rng.uniform(800, 950)
        t = self._random_start_time()

        speed = self.rng.uniform(8, 15)    # start at driving speed
        stopped = False
        stop_timer = 0

        for i in range(duration_s):
            # Traffic light simulation
            if stopped:
                stop_timer -= 1
                speed = 0
                if stop_timer <= 0:
                    stopped = False
            elif self.rng.random() < 0.005:  # ~0.5% chance of red light per second
                stopped = True
                stop_timer = int(self.rng.uniform(30, 90))
                speed = 0
            else:
                # Car speed: accelerate toward target
                target_speed = self.rng.uniform(8, 25)
                speed += (target_speed - speed) * 0.1
                speed = max(0, speed + self.rng.normal(0, 0.5))

            # HR implausibly low — sitting in car
            hr = int(self.rng.normal(72, 8))
            hr = np.clip(hr, 60, 95)

            # Cadence near 0 — not moving legs
            cadence = int(self.rng.uniform(0, 15))

            grade = self.rng.normal(0, 1.5)

            bearing = self.rng.normal(90, 5)  # follows road
            lat, lon = self._move(lat, lon, speed, bearing)
            alt += self.rng.normal(0, 0.3)

            points.append(GPSPoint(
                timestamp=t + timedelta(seconds=i),
                lat=lat, lon=lon, altitude_m=alt,
                speed_ms=speed, hr_bpm=hr,
                cadence_rpm=cadence, grade_pct=grade,
            ))

        return SyntheticActivity(
            activity_id=str(uuid.uuid4())[:8],
            anomaly_type=CAR_GPS,
            anomaly_label=ANOMALY_LABELS[CAR_GPS],
            sport_type="run",   # mislabelled — that's the anomaly
            start_time=t,
            points=points,
        )

    def generate_ebike(self) -> SyntheticActivity:
        """
        E-bike recorded as regular cycling.

        Characteristics:
        - Sustained 30-45 km/h (8-12.5 m/s) on flat terrain
        - HR low relative to speed (motor does the work)
        - Very consistent speed — no effort variation
        - Cadence looks like cycling but speed too high for effort
        """
        duration_s = int(self.rng.uniform(20 * 60, 60 * 60))

        points = []
        lat, lon = self._random_start()
        alt = self.rng.uniform(800, 950)
        t = self._random_start_time()

        base_speed = self.rng.uniform(9, 12)   # 32-43 km/h

        for i in range(duration_s):
            # Very consistent speed — motor maintains it
            speed = base_speed + self.rng.normal(0, 0.3)
            speed = max(5, speed)

            # HR low — not working hard
            # Key signal: HR vs speed inconsistency
            hr = int(self.rng.normal(105, 10))
            hr = np.clip(hr, 85, 130)

            # Cadence looks normal for cycling but speed is too high
            cadence = int(self.rng.normal(70, 8))

            grade = self.rng.normal(0, 1.0)
            # Speed barely affected by grade (motor assists)
            effective_speed = speed * max(0.85, 1.0 - grade * 0.01)

            bearing = self.rng.normal(90, 15)
            lat, lon = self._move(lat, lon, effective_speed, bearing)
            alt += self.rng.normal(0, 0.4)

            points.append(GPSPoint(
                timestamp=t + timedelta(seconds=i),
                lat=lat, lon=lon, altitude_m=alt,
                speed_ms=effective_speed, hr_bpm=hr,
                cadence_rpm=cadence, grade_pct=grade,
            ))

        return SyntheticActivity(
            activity_id=str(uuid.uuid4())[:8],
            anomaly_type=EBIKE,
            anomaly_label=ANOMALY_LABELS[EBIKE],
            sport_type="ride",
            start_time=t,
            points=points,
        )

    def generate_gps_corruption(self) -> SyntheticActivity:
        """
        Normal activity with GPS signal corruption.

        Characteristics:
        - Random teleportation jumps (GPS re-acquisition)
        - Signal dropouts (gaps in data — filled with interpolation)
        - Implausible speed spikes from position jumps
        - Inconsistent point density
        """
        duration_s = int(self.rng.uniform(30 * 60, 60 * 60))
        base_speed = self.rng.uniform(2.8, 4.0)  # underlying real run

        points = []
        lat, lon = self._random_start()
        alt = self.rng.uniform(800, 950)
        t = self._random_start_time()

        i = 0
        while i < duration_s:
            # Signal dropout: skip 10-60 seconds
            if self.rng.random() < 0.003:
                dropout = int(self.rng.uniform(10, 60))
                i += dropout
                continue

            # GPS teleportation: random position jump
            if self.rng.random() < 0.002:
                lat += self.rng.uniform(-0.01, 0.01)  # ~1km jump
                lon += self.rng.uniform(-0.01, 0.01)

            speed = max(0, base_speed + self.rng.normal(0, 0.3))

            # Occasional massive speed spike from position error
            if self.rng.random() < 0.005:
                speed = self.rng.uniform(20, 80)  # impossible speed

            hr = int(self.rng.normal(150, 10))
            hr = np.clip(hr, 110, 190)
            cadence = int(self.rng.normal(168, 8))
            grade = self.rng.normal(0, 3.0)

            bearing = self.rng.normal(90, 25)
            lat, lon = self._move(lat, lon, speed, bearing)
            alt += self.rng.normal(0, 0.8)

            points.append(GPSPoint(
                timestamp=t + timedelta(seconds=i),
                lat=lat, lon=lon, altitude_m=alt,
                speed_ms=speed, hr_bpm=hr,
                cadence_rpm=cadence, grade_pct=grade,
            ))
            i += 1

        return SyntheticActivity(
            activity_id=str(uuid.uuid4())[:8],
            anomaly_type=GPS_CORRUPT,
            anomaly_label=ANOMALY_LABELS[GPS_CORRUPT],
            sport_type="run",
            start_time=t,
            points=points,
        )

    def generate_wrong_sport(self) -> SyntheticActivity:
        """
        Cycling activity uploaded as a run.

        Characteristics:
        - Speed: 4-10 m/s (15-36 km/h) — too fast for running
        - Cadence: 75-95 rpm (cycling) vs 160-185 spm (running)
        - HR pattern consistent with cycling effort
        - Consistent speed on flat terrain (no running fatigue)
        """
        duration_s = int(self.rng.uniform(30 * 60, 90 * 60))
        base_speed = self.rng.uniform(5, 9)   # 18-32 km/h cycling

        points = []
        lat, lon = self._random_start()
        alt = self.rng.uniform(800, 950)
        t = self._random_start_time()

        for i in range(duration_s):
            fatigue = 1.0 - (i / duration_s) * 0.05
            speed = base_speed * fatigue + self.rng.normal(0, 0.4)
            speed = max(2, speed)

            hr = int(self.rng.normal(145 + speed * 3, 6))
            hr = np.clip(hr, 120, 180)

            # Cycling cadence — key distinguishing feature from running
            cadence = int(self.rng.normal(82, 6))

            grade = self.rng.normal(0, 2.5)
            speed *= max(0.7, 1.0 - grade * 0.04)

            bearing = self.rng.normal(90, 20)
            lat, lon = self._move(lat, lon, speed, bearing)
            alt += self.rng.normal(0, 0.5)

            points.append(GPSPoint(
                timestamp=t + timedelta(seconds=i),
                lat=lat, lon=lon, altitude_m=alt,
                speed_ms=speed, hr_bpm=hr,
                cadence_rpm=cadence, grade_pct=grade,
            ))

        return SyntheticActivity(
            activity_id=str(uuid.uuid4())[:8],
            anomaly_type=WRONG_SPORT,
            anomaly_label=ANOMALY_LABELS[WRONG_SPORT],
            sport_type="run",  # wrong label — should be ride
            start_time=t,
            points=points,
        )

    def generate_partial_anomaly(self) -> SyntheticActivity:
        """
        Normal run with car GPS splice at start or end.
        This is the HARDEST case — Strava specifically mentioned it.
        "Left GPS running while driving home after activity."

        Structure:
        - 10-30% of activity = car GPS (start or end)
        - 70-90% = normal run
        """
        total_duration = int(self.rng.uniform(30 * 60, 75 * 60))
        splice_pct = self.rng.uniform(0.10, 0.30)
        splice_seconds = int(total_duration * splice_pct)
        run_seconds = total_duration - splice_seconds

        # Decide whether splice is at start or end
        splice_at_start = self.rng.random() < 0.5

        points = []
        lat, lon = self._random_start()
        alt = self.rng.uniform(800, 950)
        t = self._random_start_time()
        run_speed = self.rng.uniform(2.8, 4.2)

        def car_point(i, lat, lon, alt):
            speed = self.rng.uniform(5, 20)
            hr = int(self.rng.normal(72, 8))
            cadence = int(self.rng.uniform(0, 10))
            grade = self.rng.normal(0, 1.0)
            bearing = self.rng.normal(90, 5)
            lat, lon = self._move(lat, lon, speed, bearing)
            alt += self.rng.normal(0, 0.2)
            return GPSPoint(
                timestamp=t + timedelta(seconds=i),
                lat=lat, lon=lon, altitude_m=alt,
                speed_ms=speed, hr_bpm=hr,
                cadence_rpm=cadence, grade_pct=grade,
            ), lat, lon, alt

        def run_point(i, lat, lon, alt):
            speed = run_speed + self.rng.normal(0, 0.2)
            hr = int(self.rng.normal(155, 8))
            cadence = int(self.rng.normal(168, 6))
            grade = self.rng.normal(0, 2.5)
            bearing = self.rng.normal(90, 20)
            lat, lon = self._move(lat, lon, speed, bearing)
            alt += self.rng.normal(0, 0.5)
            return GPSPoint(
                timestamp=t + timedelta(seconds=i),
                lat=lat, lon=lon, altitude_m=alt,
                speed_ms=speed, hr_bpm=hr,
                cadence_rpm=cadence, grade_pct=grade,
            ), lat, lon, alt

        if splice_at_start:
            for i in range(splice_seconds):
                p, lat, lon, alt = car_point(i, lat, lon, alt)
                points.append(p)
            for i in range(run_seconds):
                p, lat, lon, alt = run_point(
                    splice_seconds + i, lat, lon, alt)
                points.append(p)
        else:
            for i in range(run_seconds):
                p, lat, lon, alt = run_point(i, lat, lon, alt)
                points.append(p)
            for i in range(splice_seconds):
                p, lat, lon, alt = car_point(
                    run_seconds + i, lat, lon, alt)
                points.append(p)

        return SyntheticActivity(
            activity_id=str(uuid.uuid4())[:8],
            anomaly_type=PARTIAL,
            anomaly_label=ANOMALY_LABELS[PARTIAL],
            sport_type="run",
            start_time=t,
            points=points,
        )

    # ── Private helpers ───────────────────────────────────────────

    def _random_start(self) -> Tuple[float, float]:
        """Random start near Bengaluru."""
        lat = self.BASE_LAT + self.rng.uniform(-0.05, 0.05)
        lon = self.BASE_LON + self.rng.uniform(-0.05, 0.05)
        return lat, lon

    def _random_start_time(self) -> datetime:
        """Random time in the last 90 days."""
        days_ago = int(self.rng.uniform(0, 90))
        hour = int(self.rng.choice([6, 7, 8, 17, 18, 19]))
        return datetime.now() - timedelta(days=days_ago, hours=hour)

    def _move(self, lat: float, lon: float,
              speed_ms: float, bearing_deg: float) -> Tuple[float, float]:
        """
        Move a GPS coordinate by speed_ms metres in bearing direction.
        Simple flat-earth approximation — accurate enough for <50km.
        """
        distance_m = speed_ms  # 1 second at speed_ms m/s
        bearing_rad = np.radians(bearing_deg)

        # Degrees per metre at this latitude
        dlat = (distance_m * np.cos(bearing_rad)) / 111320
        dlon = (distance_m * np.sin(bearing_rad)) / (
            111320 * np.cos(np.radians(lat)))

        return lat + dlat, lon + dlon


if __name__ == "__main__":
    print("🏃 Generating synthetic GPS dataset...")
    generator = SyntheticGPSGenerator(seed=42)
    activities_df, streams_df = generator.generate_dataset(
        n_per_class=200,
        save_path="data/raw",
    )
    print("\nSample activity metadata:")
    print(activities_df.head(3).to_string())
    print("\nSample GPS stream:")
    print(streams_df.head(3).to_string())
