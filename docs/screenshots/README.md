# Screenshots for the assignment walkthrough

Save PNG captures here and reference them from [`../MLOps_Assignment2_EndToEnd.md`](../MLOps_Assignment2_EndToEnd.md).

| File | What to capture |
|------|-----------------|
| `01-ticket-profiles.png` | Editor: `REFERENCE` vs `DRIFTED` in `ticket_generation_profiles.py` |
| `02-dvc-repro.png` | Terminal: successful `dvc repro` (or `dvc dag`) |
| `03-mlflow-compare-runs.png` | MLflow: experiment `churn_model_comparison`, multiple runs |
| `04-mlflow-model-registry.png` | MLflow Models: `churn_sklearn` in **Production** |
| `05-docker-run-curl.png` | Terminal: `docker run` + `curl` to `/predict-risk` |
| `06-drift-report-json.png` | `drift_report.json` with `any_drift: true` |
| `07-github-ct-workflow.png` | GitHub Actions: **CT-on-drift** workflow (optional until you run it) |
| `tests-succesful.png` | Terminal: `pytest -q` — **16 passed** (filename spelling as saved) |

If your course forbids large binaries in Git, keep screenshots elsewhere and only link paths in your submission.
