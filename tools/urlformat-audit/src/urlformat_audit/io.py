"""Reading labelled URL datasets from CSV/JSON with flexible column and label handling."""
from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd

URL_COLS = ("url", "URL", "Url", "uri")
LABEL_COLS = ("label", "Label", "class", "Class", "ClassLabel", "type", "status", "result")


def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".json"):
        with open(path) as fh:
            data = json.load(fh)
        return pd.DataFrame({"url": data}) if isinstance(data, list) else pd.DataFrame(data)
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def resolve_columns(df: pd.DataFrame, url_col: Optional[str], label_col: Optional[str]) -> Tuple[str, Optional[str]]:
    u = url_col or next((c for c in URL_COLS if c in df.columns), None)
    if u is None:
        raise ValueError(f"no URL column found; pass --url-col (columns: {list(df.columns)})")
    l = label_col or next((c for c in LABEL_COLS if c in df.columns), None)
    return u, l


def binarize(values: pd.Series, positive: Optional[str]) -> np.ndarray:
    """Map raw labels to 0/1. ``positive`` names the positive (e.g. phishing) value; empty labels -> -1."""
    v = values.astype(str).str.strip()
    nonempty = sorted(set(v) - {""})
    if positive is None:
        if set(nonempty) <= {"0", "1", "0.0", "1.0"}:
            positive = "1"
        else:
            raise ValueError(f"cannot infer the positive label from {nonempty[:10]}; pass --positive")

    def norm(x: str) -> str:
        x = x.lower()
        try:
            return str(float(x))
        except ValueError:
            return x

    p = norm(positive)
    return np.array([-1 if x == "" else int(norm(x) == p) for x in v], dtype=int)
