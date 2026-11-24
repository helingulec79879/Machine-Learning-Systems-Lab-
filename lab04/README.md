# Lab 4 — Data pipeline and quality (UCI Adult)

## Run (one command)

```bash
pip install -r requirements.txt
python run_pipeline.py        # or: make
```

The first run downloads Adult from OpenML (internet needed, ~1 min); later runs use the cache.
Offline: put a CSV at `data/raw/adult_raw.csv` and set `data.source: csv` in `config.yaml`.

## What each lab item maps to

| Lab item | Where |
|---|---|
| 1. Problem definition | `problem_definition.md` |
| 2. Ingestion (no modification, log 48842 × 15) | `src/lab4/ingest.py` → `results/pipeline.log`, `results/ingest_log.json`, read-only `data/raw/adult_raw.csv` |
| 3. Validation (schema, types, ranges, nulls, categories, duplicates) | `src/lab4/schema.py`, `src/lab4/validate.py` → `results/validation_report.md/.json`, `results/validation_violations.csv` |
| 4. Split 70/30 stratified → impute / clip / dedup train only → joblib | `src/lab4/preprocess.py`, `src/lab4/transformers.py` → `results/preprocessor.joblib`, `results/leakage_checks.json`, `results/preprocess_summary.json` |
| 5. Derived features + LogisticRegression with/without | `FeatureEngineer` in `transformers.py` → `results/metrics.md/.json` |
| 6. Label quality (duplicate-conflict audit + 50-row re-check) | `src/lab4/label_quality.py` → `results/label_quality_report.md`, `results/label_conflicts.csv`, `results/manual_check_50.csv` |
| 7. SHA256 + config YAML + data card + one command | `data/processed/SHA256SUMS`, `results/run_config.yaml`, `data_card.md`, `run_pipeline.py` / `Makefile` |

## Leakage safeguards
- Split happens on validated-but-uncleaned data, before any statistic is computed.
- Imputers, percentile clipper, scaler and one-hot encoder live in one sklearn `Pipeline`
  that is `fit` on train only; the test set is only `transform`ed.
- Exact duplicates are removed from train only. Test rows are never dropped or changed.
- Derived features are row-wise formulas (no fitted state).
- The raw DataFrame fingerprint is re-checked at the end (`raw_data_unchanged_after_pipeline: true`).

## Reproducibility
Seed 42 everywhere, all parameters in `config.yaml`, versions + hashes in `results/run_config.yaml`.
Running twice gives the same `processed_dataset_sha256` (same library versions).
