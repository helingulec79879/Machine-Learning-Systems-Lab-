"""Step 2 - Ingestion: read raw data WITHOUT modifying it."""
import hashlib
import logging
import os
import stat

import pandas as pd

log = logging.getLogger("lab4")


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame; used to prove the raw data was never mutated."""
    h = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    cols = "|".join(map(str, df.columns)).encode()
    return hashlib.sha256(cols + h).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest(cfg: dict) -> tuple[pd.DataFrame, dict]:
    d = cfg["data"]
    if d["source"] == "openml":
        from sklearn.datasets import fetch_openml
        bunch = fetch_openml(name=d["openml_name"], version=d["openml_version"],
                             as_frame=True, data_home=d["cache_dir"])
        raw = bunch.frame
        src = f"OpenML fetch_openml(name='{d['openml_name']}', version={d['openml_version']})"
        snap = d["raw_snapshot"]
        os.makedirs(os.path.dirname(snap), exist_ok=True)
        if os.path.exists(snap):
            os.chmod(snap, stat.S_IWUSR | stat.S_IRUSR)
        raw.to_csv(snap, index=False)          # untouched copy for audit
        os.chmod(snap, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # read-only
    else:
        snap = d["csv_path"]
        raw = pd.read_csv(snap, na_values=[], keep_default_na=True)
        src = f"local CSV {snap}"

    n_rows, n_cols = raw.shape
    log.info("INGEST source=%s", src)
    log.info("INGEST row_count=%d column_count=%d", n_rows, n_cols)
    if n_rows != d["expected_rows"] or n_cols != d["expected_columns"]:
        log.warning("INGEST shape %s differs from expected (%d, %d)",
                    raw.shape, d["expected_rows"], d["expected_columns"])

    info = {
        "source": src,
        "row_count": int(n_rows),
        "column_count": int(n_cols),
        "columns": list(map(str, raw.columns)),
        "dtypes": {str(c): str(t) for c, t in raw.dtypes.items()},
        "raw_snapshot": snap,
        "raw_snapshot_sha256": file_sha256(snap),
        "raw_frame_fingerprint": frame_fingerprint(raw),
        "shape_matches_expected": bool(n_rows == d["expected_rows"] and n_cols == d["expected_columns"]),
    }
    return raw, info
