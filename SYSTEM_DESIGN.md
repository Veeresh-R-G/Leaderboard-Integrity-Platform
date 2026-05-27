# System Design — Strava Leaderboard Integrity System

> How to detect anomalous GPS activities at 85,000 uploads/minute
> using self-supervised contrastive learning.

---

## Problem Statement

Strava processes approximately **51 million activities per week** — 85,000 per minute at peak. Every activity is eligible to appear on segment leaderboards. Without integrity checking, leaderboards fill with:

- Car GPS recorded as a run or ride
- E-bikes on cycling leaderboards
- GPS corruption producing impossible speeds
- Wrong sport type classification
- Partial anomalies (real run + car GPS splice)

Strava's pre-ML rules-based system removed 6.5M activities in December 2024. Their ML system removed 4.45M anomalous runs in May 2025 and 3.9M rides in January 2026. Known remaining gaps: velodrome riding, drafting, short segments under 500m.

---

## Scale Requirements

| Metric                  | Value                       |
| ----------------------- | --------------------------- |
| Activities per week     | 51,000,000                  |
| Peak throughput         | 85,000 / minute             |
| GPS points per activity | ~2,000 (avg 30 min at 1Hz)  |
| Scoring latency target  | < 2 seconds (real-time)     |
| Backfill throughput     | 1,000,000 activities / hour |
| False positive target   | < 5% on genuine activities  |

---

## High-Level Architecture

```
                        ┌─────────────────────┐
                        │   Strava Activity   │
                        │      Upload         │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  Apache Kafka       │
                        │  Topic: activity.raw│
                        │  Partitions: 32     │
                        │  Retention: 24h     │
                        └──────────┬──────────┘
                                   │
          ┌────────────────────────┼──────────────────────┐
          │                        │                      │
┌─────────▼────────┐   ┌──────────▼───────┐   ┌─────────▼────────┐
│  Feature Worker  │   │  Feature Worker  │   │  Feature Worker  │
│  (Faust/Python)  │   │  (Faust/Python)  │   │  (Faust/Python)  │
│                  │   │                  │   │                  │
│  GPS stream pull │   │  GPS stream pull │   │  GPS stream pull │
│  Feature extract │   │  Feature extract │   │  Feature extract │
└─────────┬────────┘   └──────────┬───────┘   └─────────┬────────┘
          └────────────────────────┼──────────────────────┘
                                   │
                        ┌──────────▼──────────┐    
                        │  Kafka Topic:       │   # This is Future State Architecture
                        │  activity.features  │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   Model Inference   │
                        │   Service (FastAPI) │
                        │                     │
                        │   TS2Vec Encoder    │
                        │   (HuggingFace Hub) │
                        │                     │
                        │   Horizontal scale  │
                        │   via K8s HPA       │
                        └──────────┬──────────┘
                                   │
               ┌───────────────────┼──────────────────┐
               │                   │                  │
    ┌──────────▼──────┐  ┌─────────▼──────┐  ┌───────▼──────────┐
    │   TimescaleDB   │  │     Redis      │  │  Alert Service   │
    │  (audit trail)  │  │  (score cache) │  │ (flag + notify)  │
    │                 │  │  TTL: 1 hour   │  │                  │
    │  anomaly_scores │  │  sub-ms lookup │  │  Leaderboard API │
    └─────────────────┘  └────────────────┘  └──────────────────┘
```

---

## Component Decisions

### Message Queue — Apache Kafka

**Why Kafka over RabbitMQ or SQS:**

```
Requirement: 85,000 events/minute peak
             Replay capability (reprocess on model update)
             At-least-once delivery guarantee

Kafka advantages:
  - Partitioned by athlete_id → same athlete's activities
    always go to same partition → preserves ordering
  - Log retention: replay all events when model is updated
    (this is how Strava does backfills efficiently)
  - Consumer groups: multiple model versions can consume
    the same topic simultaneously for A/B testing
  - Throughput: 1M+ messages/second on modest hardware

Config:
  Topic: activity.raw
  Partitions: 32     (parallelism = n_partitions)
  Replication: 3     (fault tolerance)
  Retention: 24h     (replay window)
```

**Partitioning strategy:**

```python
# Key = athlete_id ensures ordering per athlete
producer.send(
    topic="activity.raw",
    key=str(athlete_id).encode(),
    value=event
)
# Same athlete's uploads go to same partition
# Prevents race conditions on leaderboard updates
```

---

### Time-Series Storage — TimescaleDB

**Why TimescaleDB over InfluxDB or plain PostgreSQL:**

```
GPS streams are time-series data:
  - 1 row per second per activity
  - 51M activities × ~2000 points = 100B rows/week
  - Need fast range queries: "get all GPS points for activity X"
  - Need compression: GPS data is highly compressible

TimescaleDB advantages over plain PostgreSQL:
  - Automatic partitioning by time (chunks)
  - 90%+ compression on GPS data via columnar storage
  - Continuous aggregates for analytics
  - Native SQL — no new query language to learn

TimescaleDB advantages over InfluxDB:
  - Full SQL joins (activities + gps_streams + anomaly_scores)
  - Better tooling ecosystem (SQLAlchemy, psycopg2)
  - ACID compliance

Compression config:
  Segment by: activity_id
  Order by: time DESC
  Compress after: 7 days
  Expected compression ratio: ~10x
```

**Hypertable chunk strategy:**

```sql
-- 1-day chunks = fast range queries within an activity
-- Activities rarely span more than 1 day
SELECT create_hypertable(
    'gps_streams', 'time',
    chunk_time_interval => INTERVAL '1 day'
);
```

---

**Inference latency breakdown:**

```
GPS stream fetch from TimescaleDB:  ~50ms
Feature normalisation:               ~5ms
Window extraction (256s windows):    ~2ms
Transformer forward pass (CPU):    ~100ms
  (GPU: ~8ms)
Redis cache write:                   ~2ms
PostgreSQL write:                   ~20ms
─────────────────────────────────────────
Total (CPU):                       ~180ms
Total (GPU):                        ~85ms

Both well within 2s target ✅
```

---

### Cache Layer — Redis

**What gets cached and why:**

```
Key:   anomaly:score:{activity_id}
Value: {score, is_flagged, reason}
TTL:   1 hour

Why Redis for this:
  Leaderboard endpoint gets called every time someone
  views a segment — high read frequency.
  Without cache: every view = TimescaleDB query.
  With cache: sub-millisecond response for cached scores.

Cache invalidation:
  On model update (backfill): delete all keys matching
  anomaly:score:* and re-score.
  Kafka replay makes this safe — no data loss.
```

---

## ML Architecture

### Why Self-Supervised Contrastive Learning

```
Strava's supervised approach requires:
  ✗ Labelled anomaly examples (expensive, requires human review)
  ✗ Retraining when new anomaly types appear
  ✗ Cannot detect what it hasn't seen (velodrome, drafting)

Our TS2Vec approach:
  ✓ Trains on normal activities only (abundant, free)
  ✓ Learns "what normal GPS movement looks like"
  ✓ Any deviation = anomaly — including unseen types
  ✓ No labelling cost
```

### TS2Vec vs TNC — Why We Use TS2Vec

```
TNC (Temporal Neighbourhood Contrastive):
  Positive pairs = windows from nearby timestamps
  Learns: nearby in time = nearby in embedding space
  Good for: sequence modelling, representation learning
  Limitation: pooled embedding loses temporal structure

TS2Vec (Temporal + Instance Contrastive):
  Two simultaneous contrastive objectives:
  1. Instance-level: different activities repel each other
  2. Temporal-level: same timestamp in two augmented views attracts

  Key innovation: applies loss at EVERY TIMESTEP
  not just the pooled representation.

  Better for anomaly detection because:
  - Partial anomalies (10% of activity is car GPS) are detected
    at the WINDOW level, not just activity level
  - A single anomalous window flags the activity via max pooling
  - TNC's pooled embedding dilutes partial anomaly signal
```

### Anomaly Scoring

```
Training:
  1. Train encoder on normal GPS activities only
  2. Compute centroid of normal activity embeddings

Inference:
  1. Extract overlapping windows (256s, 50% overlap)
  2. Encode each window → 64-dim embedding
  3. Compute L2 distance from normal centroid
  4. Score = max distance across windows (catches partial anomalies)
  5. Flag if score > threshold (calibrated on real data)

Threshold calibration:
  Set at 95th percentile of real normal activity scores.
  Means top 5% of genuine activities flagged for review.
  Acceptable false positive rate for leaderboard integrity.
```

---

## Scaling to Strava's Actual Scale

### Current Implementation (Portfolio)

```
Infrastructure:  Docker Compose (5 containers)
Throughput:      ~9 activities/second (batch backfill)
Latency:         ~180ms per activity (CPU)
Storage:         Local TimescaleDB
```

### Production Path (How We'd Scale)

```
Stage 1 — Vertical scaling (0→1M activities/day):
  Move to cloud VMs (AWS c5.4xlarge)
  GPU inference (NVIDIA T4): 10x latency improvement
  Managed Kafka (Confluent Cloud)
  Managed TimescaleDB (Timescale Cloud)
  Estimated cost: ~$800/month

Stage 2 — Horizontal scaling (1M→10M activities/day):
  Kubernetes deployment
  HPA on Kafka consumer lag metric
  Read replicas for TimescaleDB
  Redis Cluster (3 nodes)
  Estimated throughput: 10,000 activities/minute

Stage 3 — Strava scale (51M activities/week):
  32 Kafka partitions → 32 parallel consumer pods
  GPU inference farm (8× T4)
  TimescaleDB multi-node cluster
  Global Redis cluster (us-east, eu-west, ap-southeast)
  Estimated throughput: 85,000 activities/minute ✅
```

---

## Failure Modes and Mitigations

| Failure              | Impact            | Mitigation                                                  |
| -------------------- | ----------------- | ----------------------------------------------------------- |
| Kafka broker down    | Events lost       | Replication factor 3, at-least-once delivery                |
| TimescaleDB down     | Can't fetch GPS   | Redis cache serves recent scores, queue backlog             |
| Model service crash  | No new scores     | K8s restarts pod, Kafka retains messages for replay         |
| HuggingFace Hub down | Can't load model  | Local model cache (hf_hub_download caches after first load) |
| False positive spike | Athletes complain | Human review queue, easy appeal process                     |
| New anomaly type     | Missed detections | Contrastive model generalises, no retraining needed         |

---

## Data Flow Summary

```
1. Athlete uploads activity via Strava mobile app

2. Activity metadata stored in PostgreSQL (activities table)
   GPS stream stored in TimescaleDB (gps_streams hypertable)

3. Kafka event published: {activity_id, sport_type, athlete_id}

4. Feature worker consumes event:
   - Fetches GPS stream from TimescaleDB
   - Extracts 61 tabular features
   - Publishes to activity.features topic

5. Inference service consumes features:
   - Extracts 256-second windows with 50% overlap
   - TS2Vec encoder produces 64-dim embeddings per window
   - Max L2 distance from normal centroid = anomaly score
   - Score + flag stored in anomaly_scores table
   - Score cached in Redis with 1-hour TTL

6. Leaderboard API:
   - Checks Redis cache first (sub-ms)
   - Falls back to PostgreSQL (20ms)
   - Filters flagged activities from leaderboard rankings

7. Batch backfill (on model update):
   - Replay Kafka topic from beginning
   - Re-score all historical activities
   - Update leaderboard rankings retroactively
```

---

## Real-World Validation

Tested against 42 real Strava running activities:

```
Clean (correctly classified as normal): 39  (92.9%)
Flagged for review:                      3   (7.1%)

Flagged activities were legitimately unusual:
  - Different effort profile from typical training runs
  - Consistent with races or structured interval sessions

Discovery during validation:
  Strava API returns half-cadence for running activities
  (steps/min/leg, not total steps/min).
  The anomaly detector correctly flagged all runs before fix —
  cycling-range cadence on a labelled run IS anomalous.
  After cadence correction: 92.9% correctly classified.
```

---

## Comparison With Strava's Approach

| Aspect            | Strava (Current)            | This System                          |
| ----------------- | --------------------------- | ------------------------------------ |
| Method            | Supervised ML (57 features) | Self-supervised contrastive learning |
| Labels required   | Yes                         | No                                   |
| New anomaly types | Requires retraining         | Generalises automatically            |
| Partial anomalies | Rule-based detection        | Window-level TS2Vec detection        |
| Explainability    | Feature importance          | SHAP + embedding distance            |
| Backfill          | Periodic manual runs        | Event replay via Kafka               |

---

## Future Work

**Offline RL from historical trajectories (CQL/IQL)**
Train directly from Strava activity logs without synthetic data.
Eliminates simulator assumptions entirely.

**Multi-modal fusion**
Combine GPS streams with device metadata, social graph signals,
and historical athlete performance for richer anomaly detection.

**Segment-level scoring**
Score individual segments within an activity, not just the whole activity.
Addresses the "short segments under 500m" gap James mentioned.
