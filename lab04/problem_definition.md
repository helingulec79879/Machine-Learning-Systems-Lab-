# Problem Definition — Lab 4 (UCI Adult)

**Task.** Predict whether a person's annual income exceeds **$50K** from 1994 US Census attributes.
This is a **binary classification** problem: positive class `>50K`, negative class `<=50K`.

**Data.** UCI Adult, loaded with `fetch_openml(name='adult', version=2, as_frame=True)`:
48,842 rows and 15 columns.

**Inputs (14 features).**
- Numeric (6): age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week.
- Categorical (8): workclass, education, marital-status, occupation, relationship, race, sex, native-country.

**Output.** `income` ∈ {`<=50K`, `>50K`}. The classes are imbalanced (about 24% `>50K`).

**Success metrics.** **F1** on the `>50K` class (main metric, because of the imbalance: a model that
always predicts `<=50K` already gets ~76% accuracy) and **accuracy** (secondary). Both are measured
only on the held-out 30% test set. Baseline model: LogisticRegression.

**Constraints.**
1. **No leakage.** The 70/30 stratified split happens *before* any cleaning. Imputers (median / mode),
   percentile clipping, scaling and encoding are fitted on the training set only and then applied to
   the test set. Duplicates are removed from train only; the test set is never altered.
2. **Reproducible pipeline.** Raw data is ingested without modification, every parameter (seed 42,
   split ratio, percentiles) is in `config.yaml`, the fitted transformer is saved, the processed data
   is fingerprinted with SHA256, and the whole pipeline runs with one command: `python run_pipeline.py`.
3. **Transparency.** Validation reports all violations instead of silently dropping rows; known biases
   and privacy issues are documented in `data_card.md`.

**Risks / scope.** The data is old (1994) and contains sensitive attributes (sex, race), so the model
is for teaching only and must not be used for decisions about real people.
