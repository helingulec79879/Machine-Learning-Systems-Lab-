"""Step 7 - data_card.md generated from the actual run numbers."""


def build(ctx: dict) -> str:
    ing, val, lq, m, h, prep = (ctx["ingest"], ctx["validation"], ctx["label_quality"],
                               ctx["metrics"], ctx["hashes"], ctx["preprocess"])
    lab = val["label_distribution"]
    tot = sum(lab.values())
    pos = lab.get(">50K", 0)
    nulls = {c: v["total_missing"] for c, v in val["nulls"]["per_column"].items() if v["total_missing"]}
    return f"""# Data Card — UCI Adult (Census Income), processed for Lab 4

## Source
- Original: UCI Machine Learning Repository, "Adult" dataset (Becker & Kohavi, 1996),
  extracted by Barry Becker from the **1994 US Census** (Current Population Survey) database.
- Retrieved via: `{ing['source']}` (OpenML data id 1590).
- DOI: 10.24432/C5XW20

## License
- **Creative Commons Attribution 4.0 International (CC BY 4.0)** (as listed on the UCI repository).
  Reuse is allowed with attribution to the original authors.

## Size and structure
- Raw: **{ing['row_count']} rows × {ing['column_count']} columns** (14 features + 1 label).
- Features: 6 integer (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week),
  8 categorical (workclass, education, marital-status, occupation, relationship, race, sex, native-country).
- Label `income`: >50K = {pos} ({100 * pos / tot:.2f}%), <=50K = {tot - pos} ({100 * (tot - pos) / tot:.2f}%) → imbalanced.
- Missing values ('?'): {nulls}; rows with any missing: {val['nulls']['rows_with_any_missing']}.
- Exact duplicate rows in raw data: {val['duplicates']['exact_duplicate_rows_incl_label']};
  duplicate feature-groups with conflicting labels: {lq['duplicate_conflicts']['conflicting_groups']}.
- Split: 70/30 stratified, seed 42 → train {prep['train_rows_before_dedup']} rows
  ({prep['train_duplicates_dropped']} exact duplicates dropped → {prep['train_rows_after_dedup']}), test {prep['test_rows']} rows (untouched).
- Processed: {prep['n_output_features']} model features (scaled numeric + derived + one-hot categorical).

## Intended use
- Teaching / benchmarking binary classification and data-pipeline practice.
- **Not** suitable for real decisions about individuals (credit, hiring, pay).

## Known biases and limitations
- **Old data**: 1994 incomes, not inflation-adjusted; the >50K threshold means something very different today.
- **Demographic imbalance**: majority White and Male rows; ~90% United-States. Minority groups and
  most countries have very few rows, so per-group estimates are unreliable.
- **Label bias**: the >50K rate differs strongly by sex and race; a model will learn and reproduce
  these historical pay gaps (a well-known fairness benchmark for this reason).
- **Proxy variables**: `relationship` (Husband/Wife) and `marital-status` encode sex; dropping `sex`
  does not remove the bias.
- **Missingness is not random**: '?' is concentrated in workclass/occupation (often people out of work),
  so mode imputation adds bias.
- **Top-coding / sentinels**: capital-gain = 99999 is a censored value; we clip to train 1st/99th percentile.
- `fnlwgt` is a census sampling weight, not a personal attribute; it is kept only because the lab schema lists it.
- Label noise: {lq['duplicate_conflicts']['rows_in_conflicting_groups']} rows sit in identical-feature groups with conflicting labels.

## Privacy concerns
- Data is **anonymised**: no names, addresses or IDs; released publicly by the Census Bureau.
- Still contains sensitive attributes (**race, sex, marital status, native country, income**).
- Rare combinations (e.g. a small native-country + occupation + age) could allow re-identification
  when linked with other data; do not join with external personal data.
- Do not use to make decisions about real people; report results in aggregate.

## Processing summary (leakage-free)
1. Ingest raw (unchanged; read-only snapshot, fingerprint checked at the end).
2. Validate schema/types/ranges/nulls/categories/duplicates — report only, no rows dropped.
3. Split 70/30 stratified **before** cleaning.
4. Fit on train only: median (numeric) / mode (categorical) imputation, clipping to 1st/99th
   percentile, scaling, one-hot. Train exact duplicates dropped; test untouched.
5. Derived features: capital_net, hours_per_age (stateless).
6. Fitted preprocessor: `results/preprocessor.joblib`.

## Baseline (LogisticRegression, test set)
| Setting | Accuracy | F1 (>50K) |
|---|---|---|
| Without derived features | {m['without_features']['accuracy']:.4f} | {m['without_features']['f1']:.4f} |
| With derived features | {m['with_features']['accuracy']:.4f} | {m['with_features']['f1']:.4f} |

## Reproducibility
- Command: `python run_pipeline.py` (or `make`).
- Config: `config.yaml`; run record with hashes: `results/run_config.yaml`.
- SHA256 processed train: `{h['train_processed_sha256']}`
- SHA256 processed test:  `{h['test_processed_sha256']}`
- SHA256 processed dataset (train+test): `{h['processed_dataset_sha256']}`
- SHA256 raw snapshot: `{ing['raw_snapshot_sha256']}`
"""
