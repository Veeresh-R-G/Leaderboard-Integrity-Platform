-- infra/init.sql
-- ═══════════════════════════════════════════════════════════════
-- Strava Leaderboard Integrity — Database Schema
-- ═══════════════════════════════════════════════════════════════

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── Athletes ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS athletes (
    athlete_id      BIGSERIAL PRIMARY KEY,
    strava_id       BIGINT UNIQUE,
    name            TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Activities ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS activities (
    activity_id     BIGSERIAL PRIMARY KEY,
    strava_id       BIGINT UNIQUE,
    athlete_id      BIGINT REFERENCES athletes(athlete_id),
    sport_type      TEXT NOT NULL,           -- run, ride, virtualride, etc
    start_time      TIMESTAMPTZ NOT NULL,
    elapsed_time_s  INTEGER,
    distance_m      FLOAT,
    elevation_m     FLOAT,
    device_type     TEXT,
    source          TEXT DEFAULT 'strava',   -- strava / synthetic
    anomaly_type    TEXT DEFAULT NULL,       -- null = normal, else type label
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── GPS Streams (TimescaleDB hypertable) ─────────────────────
-- This is the hot path — 85k activities/min at Strava scale
-- Each activity has ~1 GPS point per second
CREATE TABLE IF NOT EXISTS gps_streams (
    time            TIMESTAMPTZ NOT NULL,
    activity_id     BIGINT NOT NULL,
    lat             DOUBLE PRECISION,
    lon             DOUBLE PRECISION,
    altitude_m      FLOAT,
    speed_ms        FLOAT,           -- speed in m/s
    hr_bpm          INTEGER,         -- heart rate
    cadence_rpm     INTEGER,         -- cadence
    power_w         INTEGER,         -- power (cycling)
    grade_pct       FLOAT            -- road grade %
);

-- Convert to hypertable (TimescaleDB magic)
-- Partitions by time — massively speeds up time-range queries
SELECT create_hypertable(
    'gps_streams', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Add compression (saves ~90% storage at Strava scale)
ALTER TABLE gps_streams SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_segmentby = 'activity_id'
);

-- Compression policy: compress chunks older than 7 days
SELECT add_compression_policy('gps_streams',
    INTERVAL '7 days', if_not_exists => TRUE);

-- ── Engineered Features (pre-computed for ML) ────────────────
CREATE TABLE IF NOT EXISTS activity_features (
    activity_id         BIGINT PRIMARY KEY REFERENCES activities(activity_id),
    -- Speed features
    max_speed_ms        FLOAT,
    mean_speed_ms       FLOAT,
    speed_p95           FLOAT,
    speed_std           FLOAT,
    pct_time_above_15ms FLOAT,   -- car threshold
    pct_time_above_8ms  FLOAT,   -- e-bike threshold
    -- Acceleration features
    max_acceleration    FLOAT,
    mean_jerk           FLOAT,
    sudden_stop_count   INTEGER,
    -- GPS quality
    gps_point_density   FLOAT,   -- points per km
    max_gap_seconds     FLOAT,
    positional_noise    FLOAT,
    -- Physiological
    hr_speed_corr       FLOAT,
    impossible_hr_speed INTEGER,
    -- Relative performance
    vs_wr_ratio         FLOAT,
    vs_athlete_pr_ratio FLOAT,
    -- Activity stats
    total_distance_km   FLOAT,
    elapsed_time_min    FLOAT,
    elevation_gain_m    FLOAT,
    computed_at         TIMESTAMPTZ DEFAULT NOW()
);

-- ── Anomaly Scores ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS anomaly_scores (
    score_id        BIGSERIAL PRIMARY KEY,
    activity_id     BIGINT REFERENCES activities(activity_id),
    model_name      TEXT NOT NULL,    -- rule_based / logistic / xgboost / contrastive
    model_version   TEXT,
    score           FLOAT NOT NULL,   -- 0.0 (normal) to 1.0 (anomalous)
    is_flagged      BOOLEAN,
    flag_reason     TEXT,
    scored_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ── Indexes ───────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_activities_athlete   ON activities(athlete_id);
CREATE INDEX IF NOT EXISTS idx_activities_sport     ON activities(sport_type);
CREATE INDEX IF NOT EXISTS idx_activities_anomaly   ON activities(anomaly_type);
CREATE INDEX IF NOT EXISTS idx_gps_activity         ON gps_streams(activity_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_scores_activity      ON anomaly_scores(activity_id);
CREATE INDEX IF NOT EXISTS idx_scores_model         ON anomaly_scores(model_name);

-- ── Useful views ──────────────────────────────────────────────
CREATE OR REPLACE VIEW activity_summary AS
SELECT
    a.activity_id,
    a.strava_id,
    a.sport_type,
    a.start_time,
    a.distance_m / 1000.0 AS distance_km,
    a.elapsed_time_s / 60.0 AS elapsed_min,
    a.anomaly_type,
    f.max_speed_ms,
    f.speed_p95,
    s.score AS latest_score,
    s.is_flagged,
    s.model_name
FROM activities a
LEFT JOIN activity_features f USING (activity_id)
LEFT JOIN LATERAL (
    SELECT score, is_flagged, model_name
    FROM anomaly_scores
    WHERE activity_id = a.activity_id
    ORDER BY scored_at DESC LIMIT 1
) s ON true;