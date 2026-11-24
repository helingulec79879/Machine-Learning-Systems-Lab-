# Lab 4 — Data pipeline and quality

UCI Adult via `fetch_openml(name='adult', version=2, as_frame=True)`, 48,842 rows × 15 columns.
70/30 stratified split, seed 42. Baseline: LogisticRegression (max_iter 2000).

Code: `run_pipeline.py` (entry point) and `src/lab4/` (`ingest.py`, `schema.py`, `validate.py`,
`preprocess.py`, `transformers.py`, `label_quality.py`, `datacard.py`). All parameters: `config.yaml`.

**One command:** `cd lab04 && pip install -r requirements.txt && python run_pipeline.py` (or `make`).
The first run downloads the data from OpenML; all outputs go to `results/`, `data/processed/` and `data_card.md`.

> **Status:** pipeline is complete. Numbers marked _see results_ are written by running the command above.

## 1. Problem definition
See [`problem_definition.md`](problem_definition.md). Binary classification of income >50K from 14 features;
main metric F1 on >50K (imbalanced, ~24% positive), secondary accuracy; constraints: no leakage, reproducible.

## 2. Ingestion
Raw data is read and saved unchanged as a read-only snapshot (`data/raw/adult_raw.csv`). Row and column counts
are logged (`results/pipeline.log`: `INGEST row_count=48842 column_count=15`). A content fingerprint of the raw
DataFrame is checked again at the end of the run to prove it was not modified.

## 3. Validation
Schema from the lab sheet (`src/lab4/schema.py`): 6 int columns, 9 category columns, ranges age 17–90,
hours-per-week 1–99, capital-gain 0–99999, capital-loss 0–4356, and '?' = missing (allowed category).
Checks: column presence, types, ranges, missing values, allowed categories, duplicates, label distribution,
education/education-num consistency. **No rows are dropped** — violations are reported in
`results/validation_report.md` and row by row in `results/validation_violations.csv`.

Known missing values ('?'): workclass 2,799, occupation 2,809, native-country 857 (3,620 rows with any missing).

## 4. Cleaning and transformation (leakage-free)
1. Split 70/30 stratified **before** cleaning → train 34,189, test 14,653.
2. Exact duplicates dropped from train only (count in `results/preprocess_summary.json`); test untouched.
3. One sklearn `Pipeline` fitted on train only: median imputer (numeric), mode imputer (categorical),
   clipping to train 1st/99th percentile, StandardScaler, one-hot encoder.
4. Saved with `joblib.dump` to `results/preprocessor.joblib` (reloaded and checked in the same run).

Evidence: `results/leakage_checks.json` (`fit_on_test_rows: 0`, `train_test_index_overlap: 0`,
`raw_data_unchanged_after_pipeline: true`).

## 5. Derived features
- `capital_net = capital-gain − capital-loss`
- `hours_per_age = hours-per-week / (age − 16)`

Both are row-wise formulas with no fitted state, so they cannot leak test information.

| Setting | Accuracy | F1 (>50K) |
|---|---|---|
| Without derived features | _see results/metrics.md_ | _see results/metrics.md_ |
| With derived features | _see results/metrics.md_ | _see results/metrics.md_ |

## 6. Label quality
- **Duplicate-conflict audit:** rows with identical 14 features but different income labels →
  count in `results/label_quality_report.md`, details in `results/label_conflicts.csv`.
- **Manual re-check:** 50 random rows (seed 42) in `results/manual_check_50.csv`, with automatic flags
  against the data dictionary (ranges, education/education-num mismatch, Husband+Female, Wife+Male,
  capital-gain 99999 sentinel, etc.) and a `manual_verdict` column for the manual check.

## 7. Hash, config, data card
- SHA256 of processed train/test: `data/processed/SHA256SUMS`; combined hash, config, library versions
  and metrics: `results/run_config.yaml`.
- Data card: [`data_card.md`](data_card.md) (source, license CC BY 4.0, size, known biases, privacy concerns).
