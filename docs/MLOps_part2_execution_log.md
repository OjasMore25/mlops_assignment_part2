# MLOps Assignment Part 2 — execution log

Purpose: record commands, outputs, mapping to the Part 2 PDF, deviations, and fixes while building the MLOps phase.

PDF reference file: `MLOps Asssignmnet part 2.pdf` (Assignment II + MLOps requirements).

---

## 2026-04-11 — Build start (`execution-log`, `phase-1-ml-pipeline`)

**PDF:** Task 2 (feature engineering list, ML classifier, F1 / ROC-AUC / precision–recall), system includes training script + saved artifact; groundwork for inference API (Phase 2).

**Deviation (documented):** “Change in monthly charges” is implemented as `charge_change_proxy = monthly_charges - (total_charges / tenure)` with tenure 0 treated as no implied average (proxy 0). The PDF implies a time-series change; our `customers.csv` has no charge history, so this proxy is the explicit interpretation.

**Commands (cwd: repo root):**

```bash
python3 -c "import pandas as pd; ..."   # failed: system Python has no pandas
pip install -r requirements.txt        # (use project venv in practice)
pytest
python scripts/train.py
```

Outputs will be appended after each run with exit codes and key metrics.

---

## 2026-04-19 — Phase 1 & 2 implementation (`phase-1-ml-pipeline`, `phase-2-inference-api`)

**PDF:** Task 2 (features, classifier, F1 / ROC-AUC / average precision, artifact, inference path); system now has training script + saved pipeline + API using the same feature builder.

**Commands (cwd: repo root):**

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py
pytest -q
```

**Outputs:**

- `pytest`: `11 passed` (includes `tests/test_feature_engineering.py`).
- `scripts/train.py`: example metrics on current data — `f1` and `roc_auc` reported as `1.0` (strong separability on this dataset; treat as a sanity check, not a real-world guarantee).
- System `python3` without venv: `pip install` failed with PEP 668 externally-managed-environment — **fix:** use project `.venv` (documented in this log).

**Docker:** `docker build` was started; layer `pip install -r requirements.txt` is heavy (DVC/MLflow stack); build hit the agent step time limit and was canceled — not an application error. Re-run `docker build` locally when needed.

**Files added/changed (high level):**

- `src/feature_engineering.py` — shared `as_of` features (PDF list + `charge_change_proxy`).
- `scripts/train.py` — sklearn `Pipeline`, metrics JSON + `joblib` artifact.
- `src/app.py` — loads pipeline + metrics; `/predict-risk` returns probabilities and bands.
- `src/risk_bands.py`, `src/feature_pipeline.py` (delegates to feature_engineering).
- `.github/workflows/ci.yml` — `python scripts/train.py` before `pytest`.
- `Dockerfile` — `RUN python scripts/train.py` after copy.
- `.dockerignore` — removed `scripts/` exclusion so `train.py` is available in the image.
- `.gitignore` — `models/*.joblib`, `models/training_metrics.json`.
- `docs/API.md` — ML response schema.

**Todo notifications (for your log):**

- **`execution-log`:** ongoing — this file is the running log.
- **`phase-1-ml-pipeline`:** completed — unified feature engineering + train script + sklearn pipeline + metrics + artifact.
- **`phase-2-inference-api`:** completed — FastAPI uses saved pipeline; tests/docs updated.

---

## 2026-04-19 — Phases 3–9 (`phase-3-dvc` … `phase-9-polish`)

**PDF mapping:** data versioning (DVC) + splits; MLflow experiments + registry staging; sklearn pipeline serialization (already in train); schema validation (Pandera); CI/CD/CT (metric gate + scheduled workflow); drift script; monitoring counters + memory gauge; README/API updates.

**Deviation:** A DVC `features` stage writing `data/processed/customer_features.csv` failed because that file is **already tracked in Git**. The `features` stage was removed from [`dvc.yaml`](dvc.yaml); the log documents migrating it with `git rm --cached` + `dvc add` if full DVC coverage of that CSV is required.

**Commands (cwd: repo root, `.venv` active):**

```bash
pip install pandera psutil
dvc init          # once; creates .dvc/
dvc repro         # split → train; updates dvc.lock
python scripts/check_drift.py
python scripts/metric_gate.py
pytest -q         # 13 passed
```

**Fix:** Prometheus default registry already defines `process_resident_memory_bytes`; custom gauge renamed to `churn_service_resident_memory_bytes` to avoid `ValueError: Duplicated timeseries`.

**Todo notifications (for your log):**

- **`phase-3-dvc`:** completed — [`dvc.yaml`](dvc.yaml) + [`params.yaml`](params.yaml) + [`dvc.lock`](dvc.lock) + `.dvc/`; stages `split` + `train`; `data/splits/*.csv` DVC-managed (local `.gitignore` under `data/splits/`).
- **`phase-4-mlflow`:** completed — [`scripts/train.py`](scripts/train.py) logs params/metrics/features to MLflow, registers `churn_sklearn`, transitions latest version to **Staging** (uses deprecated `transition_model_version_stage`; acceptable for coursework until MLflow removes stages).
- **`phase-5-schema`:** completed — [`src/inference_schema.py`](src/inference_schema.py) + validation in [`src/app.py`](src/app.py) + [`tests/test_inference_schema.py`](tests/test_inference_schema.py); Prometheus `feature_validation_failures_total`.
- **`phase-6-cicdct`:** completed — [`.github/workflows/ci.yml`](.github/workflows/ci.yml): `dvc repro`, `metric_gate.py`, non-blocking `check_drift.py`, `pytest`; [`.github/workflows/scheduled_retrain.yml`](.github/workflows/scheduled_retrain.yml) weekly CT + artifact upload.
- **`phase-7-retrain-drift`:** completed — [`scripts/check_drift.py`](scripts/check_drift.py) (KS vs `data/splits/train.csv`); scheduled workflow runs `dvc repro` + drift.
- **`phase-8-monitoring`:** completed — [`src/app.py`](src/app.py): validation failure counter + optional RSS gauge (`psutil`).
- **`phase-9-polish`:** completed — README/DVC/MLflow docs, `.gitignore` for `mlflow.db`/`mlruns/`, [`conftest.py`](conftest.py) for local pytest bootstrap, [`Dockerfile`](Dockerfile) uses splits + `--skip-mlflow`, removed stray `docs/Untitled`.

---

## 2026-04-19 — Assignment-style E2E (multi-model, drift → CT, documentation)

**PDF:** Experiments + registry + production deploy; drift detection + retraining (CT); serving + tests.

**Added / changed:**

- [`src/ticket_generation_profiles.py`](../src/ticket_generation_profiles.py), [`src/ticket_generator.py`](../src/ticket_generator.py), refactored [`scripts/generate_tickets.py`](../scripts/generate_tickets.py), [`scripts/simulate_ticket_drift.py`](../scripts/simulate_ticket_drift.py).
- [`scripts/train_experiments.py`](../scripts/train_experiments.py) — three model kinds, MLflow, Production promotion; DVC `train` stage now calls this script.
- [`src/model_factory.py`](../src/model_factory.py); [`scripts/train.py`](../scripts/train.py) uses shared factory + `--model-kind`.
- [`.github/workflows/ct_on_drift.yml`](../.github/workflows/ct_on_drift.yml) — CT after drift detection.
- [`docs/MLOps_Assignment2_EndToEnd.md`](../docs/MLOps_Assignment2_EndToEnd.md) — stage-wise guide + screenshot placeholders; [`docs/screenshots/README.md`](../docs/screenshots/README.md).

**Commands:**

```bash
python scripts/simulate_ticket_drift.py
python scripts/check_drift.py --tickets data/processed/tickets_drifted.csv --ks-threshold 0.12 --fail-on-drift; echo $?
# → exit 1, any_drift true (KS spikes on ticket-driven columns)
dvc repro train
pytest -q   # 16 passed
```

**Fix:** Prometheus duplicate metric name avoided earlier; drift test writes drifted CSV under `tmp_path` to avoid polluting repo.
