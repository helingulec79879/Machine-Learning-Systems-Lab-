#!/usr/bin/env python3
"""Lab 4 — one-command data pipeline for UCI Adult.

    python run_pipeline.py              # full run (downloads from OpenML on first run)
    python run_pipeline.py --config config.yaml
"""
import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from lab4 import datacard, label_quality, preprocess, schema as S, validate
from lab4.ingest import file_sha256, frame_fingerprint, ingest

log = logging.getLogger("lab4")


def setup_logging(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(results_dir, "pipeline.log"), mode="w")):
        h.setFormatter(fmt)
        log.addHandler(h)


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def evaluate(cfg, Xtr, ytr, Xte, yte, with_features):
    pre = preprocess.build_preprocessor(cfg, with_features=with_features)
    mc = cfg["model"]
    model = Pipeline([("pre", pre),
                      ("clf", LogisticRegression(max_iter=mc["max_iter"], C=mc["C"],
                                                 random_state=mc["random_state"]))])
    model.fit(Xtr, ytr)                       # fit on TRAIN only
    pred = model.predict(Xte)
    return model, {"accuracy": float(accuracy_score(yte, pred)),
                   "f1": float(f1_score(yte, pred, pos_label=1)),
                   "n_features": int(len(pre.get_feature_names_out()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # relative paths = lab04/
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    np.random.seed(cfg["split"]["random_state"])
    R = cfg["output"]["results_dir"]
    P = cfg["output"]["processed_dir"]
    os.makedirs(P, exist_ok=True)
    setup_logging(R)
    t0 = time.time()

    # ---- 2. Ingestion ---------------------------------------------------------
    raw, ing = ingest(cfg)
    dump_json(ing, f"{R}/ingest_log.json")

    # ---- 3. Validation (on a standardised COPY; nothing dropped) --------------
    df, notes = validate.standardize(raw, cfg)
    vrep, vdf = validate.validate(df, ing["dtypes"])
    vrep["standardisation_notes"] = notes
    dump_json(vrep, f"{R}/validation_report.json")
    write(f"{R}/validation_report.md", validate.report_to_markdown(vrep, notes))
    vdf.to_csv(f"{R}/validation_violations.csv", index=False)
    assert len(df) == len(raw), "validation must not drop rows"

    # ---- 6. Label quality (read-only audit on full data) ----------------------
    dup, conflicts = label_quality.duplicate_conflicts(df)
    lq = cfg["label_quality"]
    man, sample = label_quality.manual_recheck(df, lq["sample_size"], lq["random_state"])
    conflicts.to_csv(f"{R}/label_conflicts.csv", index=False)
    sample.to_csv(f"{R}/manual_check_50.csv", index=False)
    write(f"{R}/label_quality_report.md", label_quality.to_markdown(dup, man))
    lqrep = {"duplicate_conflicts": dup, "manual_recheck": man}
    dump_json(lqrep, f"{R}/label_quality_report.json")

    # ---- 4. Split BEFORE cleaning, then clean train only ----------------------
    train, test = preprocess.split(df, cfg)
    assert set(train.index).isdisjoint(test.index)
    n_train_raw = len(train)
    n_dup = 0
    if cfg["cleaning"]["drop_train_duplicates"]:
        train, n_dup = preprocess.drop_train_duplicates(train)
    Xtr, ytr = preprocess.to_model_frame(train, cfg["cleaning"]["missing_token"])
    Xte, yte = preprocess.to_model_frame(test, cfg["cleaning"]["missing_token"])

    # ---- 5. Baseline with / without derived features --------------------------
    _, m_without = evaluate(cfg, Xtr, ytr, Xte, yte, with_features=False)
    model_with, m_with = evaluate(cfg, Xtr, ytr, Xte, yte, with_features=True)
    metrics = {"model": "LogisticRegression", "positive_class": ">50K",
               "without_features": m_without, "with_features": m_with,
               "delta_accuracy": m_with["accuracy"] - m_without["accuracy"],
               "delta_f1": m_with["f1"] - m_without["f1"]}
    dump_json(metrics, f"{R}/metrics.json")
    write(f"{R}/metrics.md",
          "| Setting | Accuracy | F1 (>50K) | # features |\n|---|---|---|---|\n"
          f"| Without derived | {m_without['accuracy']:.4f} | {m_without['f1']:.4f} | {m_without['n_features']} |\n"
          f"| With capital_net + hours_per_age | {m_with['accuracy']:.4f} | {m_with['f1']:.4f} | {m_with['n_features']} |\n"
          f"| Δ (with − without) | {metrics['delta_accuracy']:+.4f} | {metrics['delta_f1']:+.4f} | |\n")
    log.info("METRICS without acc=%.4f f1=%.4f | with acc=%.4f f1=%.4f",
             m_without["accuracy"], m_without["f1"], m_with["accuracy"], m_with["f1"])

    # Fitted preprocessor (train-only) -> results/preprocessor.joblib
    pre = model_with.named_steps["pre"]
    joblib.dump(pre, cfg["output"]["preprocessor_path"])
    reloaded = joblib.load(cfg["output"]["preprocessor_path"])
    assert np.allclose(reloaded.transform(Xte.head(50)), pre.transform(Xte.head(50)))
    psum = preprocess.fitted_summary(pre)
    prep = {"train_rows_before_dedup": n_train_raw, "train_duplicates_dropped": n_dup,
            "train_rows_after_dedup": len(train), "test_rows": len(test), **psum}
    dump_json(prep, f"{R}/preprocess_summary.json")

    # ---- 7. Processed dataset + SHA256 ----------------------------------------
    cols = list(pre.get_feature_names_out())
    tr = pd.DataFrame(pre.transform(Xtr), columns=cols)
    tr["income"] = ytr.values
    te = pd.DataFrame(pre.transform(Xte), columns=cols)
    te["income"] = yte.values
    trp, tep = f"{P}/train_processed.csv", f"{P}/test_processed.csv"
    tr.to_csv(trp, index=False, float_format="%.8f", lineterminator="\n")
    te.to_csv(tep, index=False, float_format="%.8f", lineterminator="\n")
    h_all = hashlib.sha256()
    for p in (trp, tep):
        with open(p, "rb") as f:
            h_all.update(f.read())
    hashes = {"train_processed_sha256": file_sha256(trp), "test_processed_sha256": file_sha256(tep),
              "processed_dataset_sha256": h_all.hexdigest(),
              "preprocessor_joblib_sha256": file_sha256(cfg["output"]["preprocessor_path"])}
    write(f"{P}/SHA256SUMS",
          f"{hashes['train_processed_sha256']}  train_processed.csv\n"
          f"{hashes['test_processed_sha256']}  test_processed.csv\n")

    # Leakage / integrity evidence
    raw_unchanged = frame_fingerprint(raw) == ing["raw_frame_fingerprint"]
    leakage = {
        "split_before_cleaning": True,
        "train_test_index_overlap": 0,
        "imputer_clipper_scaler_encoder_fit_rows": len(train),
        "fit_on_test_rows": 0,
        "test_duplicates_removed": 0,
        "derived_features_stateless": True,
        "raw_data_unchanged_after_pipeline": bool(raw_unchanged),
    }
    assert raw_unchanged, "raw data was modified!"
    dump_json(leakage, f"{R}/leakage_checks.json")

    run_record = {
        "config": cfg,
        "hashes": {**hashes, "raw_snapshot_sha256": ing["raw_snapshot_sha256"]},
        "data": {"raw_rows": ing["row_count"], "raw_cols": ing["column_count"],
                 "train_rows": len(train), "test_rows": len(test),
                 "processed_columns": len(cols) + 1},
        "metrics": metrics,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(f"{R}/run_config.yaml", "w") as f:
        yaml.safe_dump(run_record, f, sort_keys=False)

    write("data_card.md", datacard.build({"ingest": ing, "validation": vrep, "label_quality": lqrep,
                                           "metrics": metrics, "hashes": hashes, "preprocess": prep}))
    log.info("HASH processed_dataset_sha256=%s", hashes["processed_dataset_sha256"])
    log.info("DONE in %.1fs — see %s/ and data_card.md", time.time() - t0, R)


if __name__ == "__main__":
    main()
