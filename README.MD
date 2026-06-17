# 🏆 Strava Leaderboard Integrity System

[Kaggle Notebook](https://www.kaggle.com/code/veeresh1104/tnc-vs-ts2ecv-comparison)

> End-to-end production-grade anomaly detection for GPS sports data.
> Self-supervised contrastive learning detects cheating on Strava leaderboards
> without requiring any labelled anomaly data.

**Reference:** Yue et al. — _TS2Vec: Towards Universal Representation of Time Series_ — AAAI 2022

---

## The Problem

Strava removed **4.45 million anomalous run activities** in May 2025 and **3.9 million ride activities** in January 2026. Their engineering blog documents known remaining gaps: velodrome riding, drafting, short segments under 500m, and partial anomalies (real run + car GPS splice).

Their current system uses supervised ML — which requires labelled anomaly examples and cannot detect anomaly types it has never seen.

**This project builds a self-supervised alternative** that learns only from normal activities and generalises to unseen anomaly types automatically.

---

## Architecture

```
Strava Upload → Kafka → TimescaleDB → Feature Extraction
                                            ↓
                                   TS2Vec Encoder
                                            ↓
                              Anomaly Score → Redis Cache
                                            ↓
                                   FastAPI → Leaderboard
```

Full system design: [SYSTEM_DESIGN.md](./SYSTEM_DESIGN.md)

---

## What's Built

### Phase 1 — Data Infrastructure

- **Apache Kafka** — activity event streaming (simulates 85k/min)
- **TimescaleDB** — GPS time-series storage with hypertable compression
- **Redis** — sub-millisecond anomaly score cache
- **Strava API ingestion** — pulls real activity history + GPS streams

### Phase 2 — Feature Engineering

61 features across 6 groups:

| Group         | Features                               | Key Signal                            |
| ------------- | -------------------------------------- | ------------------------------------- |
| Speed         | mean, max, p95, p99, stops, CV         | Car GPS has high mean + high variance |
| Acceleration  | jerk, sharp braking, superhuman events | Cars brake sharply at traffic lights  |
| GPS quality   | dropouts, spikes, point density        | Corruption produces impossible speeds |
| Physiological | HR-speed correlation                   | Car: HR 72 at 45 km/h = impossible    |
| Cadence       | run zone %, ride zone %, ratio         | Cycling cadence on a "run"            |
| Grade/terrain | hill speed ratio                       | Cars barely slow on hills             |

### Phase 3 — Baseline Models

| Model                      | ROC-AUC | F1    | Needs Labels? |
| -------------------------- | ------- | ----- | ------------- |
| Rule-Based (Strava pre-ML) | 1.000   | 0.977 | No            |
| Logistic Regression        | 1.000   | 1.000 | **Yes**       |
| XGBoost + SHAP             | 1.000   | 0.998 | **Yes**       |

> Callout : We used sythetic data while training, hence the perfect score in the metric

### Phase 4 — Self-Supervised Contrastive Learning

Two architectures compared:

**TNC (Temporal Neighbourhood Contrastive)**
Nearby GPS windows in time → similar embeddings.
Learns temporal continuity of physiological state.

**TS2Vec (Temporal + Instance Contrastive)**
Two simultaneous objectives:

- Instance-level: different activities repel in embedding space
- Temporal-level: same timestamp in two augmented views attracts

TS2Vec applies the loss at **every timestep** — not just the pooled representation. This makes it better at detecting partial anomalies (where only part of the activity is anomalous).

| Model   | ROC-AUC | F1    | Needs Labels? | Partial Anomalies |
| ------- | ------- | ----- | ------------- | ----------------- |
| XGBoost | 1.000   | 0.998 | Yes           | Weak              |
| TNC     | 1.000   | 0.980 | **No**        | Moderate          |
| TS2Vec  | 1.000   | 0.980 | **No**        | **Strong**        |

### Phase 5 — Inference Pipeline

- **FastAPI** inference service
- **Kafka consumer** for real-time scoring
- **Batch backfill** for retroactive leaderboard cleaning
- **Real Strava validation** on 42 personal activities (12 Weeks Activities, Yes I had a deload phase 👀)

---

## Anomaly Types Detected

| Type             | How Detected                                   |
| ---------------- | ---------------------------------------------- |
| Car GPS          | High speed + low HR + sudden stops             |
| E-bike           | Sustained 35+ km/h + low HR + consistent speed |
| GPS corruption   | Impossible speed spikes (>80 m/s)              |
| Wrong sport type | Cycling cadence on a run label                 |
| Partial anomaly  | TS2Vec window-level detection                  |

---

## Real-World Validation

Tested against **42 real Strava running activities**:

```
Clean (correctly normal): 39 (92.9%)
Flagged for review:        3  (7.1%)

Flagged activities were legitimately unusual:
  - Different effort profile from typical training runs
  - Consistent with race or structured interval sessions
```

**Real engineering discovery during validation:**

Strava's API returns half-cadence for running activities (steps/min per leg, not total). The anomaly detector correctly flagged all runs before the fix — cycling-range cadence on a labelled run IS anomalous. After cadence correction: 92.9% correctly classified.

---

## UMAP Embedding Visualisation

Both TNC and TS2Vec learn to separate anomaly types in embedding space without any labels:

- **TNC** produces linear manifolds (sequential structure preserved)
- **TS2Vec** produces compact clusters (invariant representations)

TS2Vec's compact clusters are better for anomaly detection — each anomaly type collapses to a distinct region regardless of which part of the activity the window came from.

---

## Quick Start

```bash
# Clone repo
git clone <repo-url>
cd leaderboard-integrity

# Start infrastructure
docker-compose up -d

# Install dependencies
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Ingest your Strava data
python ingestion/strava/strava_client.py

# Run batch scoring
python ingestion/batch_backfill.py

# Start inference API
uvicorn api.main:app --reload --port 8001
```

---

## API Endpoints

```bash
# Score a GPS activity
POST /score
{
  "activity_id": "12345",
  "gps_points": [
    {"speed_ms": 3.2, "hr_bpm": 155, "cadence_rpm": 168, "grade_pct": 0.5}
  ]
}

# Get cached score
GET /activity/{activity_id}/score

# Pipeline stats
GET /stats
```

---

## Tech Stack

| Component       | Technology                                    |
| --------------- | --------------------------------------------- |
| Event streaming | Apache Kafka (Confluent)                      |
| Time-series DB  | TimescaleDB (PostgreSQL extension)            |
| Cache           | Redis                                         |
| ML framework    | PyTorch                                       |
| Inference API   | FastAPI                                       |
| Orchestration   | Docker Compose → Kubernetes (production path) |
| Notebooks       | Kaggle (free GPU)                             |

---

## System Design Highlights

**Why Kafka over SQS:**
Kafka's log retention enables full replay when the model is updated — this is how Strava executes their backfill runs efficiently.

**Why TimescaleDB over InfluxDB:**
Full SQL joins across activities, GPS streams, and anomaly scores. 90%+ compression on GPS data via columnar storage.

**Why TS2Vec over SimCLR:**
SimCLR was designed for images. TS2Vec applies contrastive loss at every timestep — preserving temporal structure critical for detecting partial anomalies in GPS time-series.

**Why self-supervised over supervised:**
Strava's supervised model cannot detect anomaly types it has never seen (velodrome, drafting). Our contrastive model learns the distribution of normal movement — anything outside that distribution is anomalous, regardless of type.

---

## References

1. Yue et al. (2022) — _TS2Vec: Towards Universal Representation of Time Series_ — AAAI 2022
2. Tonekaboni et al. (2021) — _Unsupervised Representation Learning for Time Series_ — NeurIPS 2021
3. Strava Engineering Blog — _Keeping Strava's Segment Leaderboards Fair_ — March 2026
4. Gabbett (2016) — _The training-injury prevention paradox_ — BJSM
5. Impellizzeri et al. (2020) — _ACWR: Conceptual Issues and Misuse_ — BJSM

---

## Project Context

A lot of leaderboards on strava do have malicious activities with wrong tags.
To tackle issues related to leaderboard integrity, this project was created.

Related projects:

- [DRL Training Load Optimizer](https://github.com/Veeresh-R-G/DRL-Training-Load-Optimizer) — PPO + Transformer for adaptive running coaching
- [DAG Training Planner](https://github.com/Veeresh-R-G/API_DAG-Based-Workout-trainer) — Constraint satisfaction on a DAG for training plan generation
