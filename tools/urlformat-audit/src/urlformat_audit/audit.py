"""Class-conditional format audit and format-only baselines for labelled URL datasets."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .parse import format_features

TEMPLATE_REGEX = r"^https://www\.[a-z0-9.-]+$"


# ----------------------------------------------------------------------------- metrics
def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Accuracy, balanced accuracy, F1 of the positive class (1) and Matthews correlation."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    tp = float(np.sum((p == 1) & (y == 1))); tn = float(np.sum((p == 0) & (y == 0)))
    fp = float(np.sum((p == 1) & (y == 0))); fn = float(np.sum((p == 0) & (y == 1)))
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "accuracy": (tp + tn) / max(len(y), 1),
        "balanced_accuracy": (tpr + tnr) / 2,
        "f1_positive": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "mcc": (tp * tn - fp * fn) / den if den > 0 else 0.0,
        "tpr": tpr, "fpr": 1 - tnr,
    }


def normalized_mutual_information(a: Sequence, b: Sequence) -> float:
    """I(A;B) / H(B): share of the label entropy explained by the format cell (0 = none, 1 = all)."""
    t = pd.crosstab(pd.Series(a, name="a"), pd.Series(b, name="b")).to_numpy(dtype=float)
    n = t.sum()
    if n == 0:
        return float("nan")
    pab = t / n; pa = pab.sum(1, keepdims=True); pb = pab.sum(0, keepdims=True)
    nz = pab > 0
    mi = float(np.sum(pab[nz] * np.log(pab[nz] / (pa @ pb)[nz])))
    hb = float(-np.sum(pb[pb > 0] * np.log(pb[pb > 0])))
    return mi / hb if hb > 0 else float("nan")


# ----------------------------------------------------------------------------- format-only rule
@dataclass
class FormatRule:
    """Predicts the majority label of each format cell seen in training (fallback: overall majority)."""
    cell_label: Dict[str, int] = field(default_factory=dict)
    default: int = 1

    @classmethod
    def fit(cls, urls: Iterable[str], labels: Sequence[int]) -> "FormatRule":
        f = format_features(urls)
        y = np.asarray(labels).astype(int)
        d = pd.DataFrame({"cell": f["cell"].values, "y": y})
        maj = d.groupby("cell")["y"].mean()
        return cls({c: int(v >= 0.5) for c, v in maj.items()}, int(y.mean() >= 0.5))

    def predict(self, urls: Iterable[str]) -> np.ndarray:
        f = format_features(urls)
        return np.array([self.cell_label.get(c, self.default) for c in f["cell"]], dtype=int)


# ----------------------------------------------------------------------------- audit
@dataclass
class AuditResult:
    name: str
    n: int
    class_table: pd.DataFrame          # % of URLs per format property, per class
    cell_table: pd.DataFrame           # rows per format cell: counts per class and purity
    format_rule: Dict[str, float]      # in-sample metrics of the majority-per-cell rule
    template_rule: Dict[str, float]    # metrics of 'legitimate iff URL matches TEMPLATE_REGEX'
    trivial: Dict[str, float]          # metrics of predicting the positive class everywhere
    nmi_cell_label: float
    flags: list

    def to_dict(self) -> dict:
        return {"name": self.name, "n": self.n, "format_rule": self.format_rule, "template_rule": self.template_rule,
                "trivial": self.trivial, "nmi_cell_label": self.nmi_cell_label, "flags": self.flags,
                "class_table": self.class_table.reset_index().to_dict(orient="records"),
                "cell_table": self.cell_table.reset_index().to_dict(orient="records")}


def audit(urls: Sequence[str], labels: Sequence[int], name: str = "dataset", flag_gap: float = 30.0) -> AuditResult:
    """Audit a labelled URL dataset. ``labels``: 1 = positive (e.g. phishing), 0 = negative (legitimate).

    flag_gap: a format property is flagged when its prevalence differs between the classes by at least this many
    percentage points.
    """
    urls = [str(u) for u in urls]
    y = np.asarray(labels).astype(int)
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("labels must be 0/1 (1 = positive class)")
    f = format_features(urls)
    f["label"] = y
    props = ["scheme=https", "scheme=http", "scheme=none", "scheme=other", "www", "trailing_slash", "path", "query", "port", "ip_host", "uppercase"]
    rows = {}
    for cls_, nm in [(0, "negative (0)"), (1, "positive (1)")]:
        g = f[f.label == cls_]
        r = {"n": len(g)}
        for p in props:
            if p.startswith("scheme="):
                r[p] = 100 * float((g["scheme"] == p.split("=")[1]).mean()) if len(g) else float("nan")
            else:
                r[p] = 100 * float(g[p].mean()) if len(g) else float("nan")
        r["median_length"] = float(g["length"].median()) if len(g) else float("nan")
        rows[nm] = r
    class_table = pd.DataFrame(rows).T

    ct = pd.crosstab(f["cell"], f["label"]).reindex(columns=[0, 1], fill_value=0)
    ct.columns = ["n_negative", "n_positive"]
    ct["n"] = ct.sum(axis=1)
    ct["share_positive"] = ct["n_positive"] / ct["n"]
    ct["purity"] = np.maximum(ct["share_positive"], 1 - ct["share_positive"])
    ct = ct.sort_values("n", ascending=False)

    rule = FormatRule.fit(urls, y)
    fmt = binary_metrics(y, rule.predict(urls))
    tpl_pred = (~pd.Series(urls).str.strip().str.match(TEMPLATE_REGEX)).astype(int).values
    tpl = binary_metrics(y, tpl_pred)
    triv = binary_metrics(y, np.ones(len(y), int))
    nmi = normalized_mutual_information(f["cell"], y)

    flags = []
    for p in props:
        gap = abs(class_table.loc["positive (1)", p] - class_table.loc["negative (0)", p])
        if gap >= flag_gap:
            flags.append(f"{p}: {class_table.loc['negative (0)', p]:.1f}% of negatives vs {class_table.loc['positive (1)', p]:.1f}% of positives")
    if fmt["mcc"] >= 0.5:
        flags.append(f"format-only rule (4 cues, 24 cells) reaches MCC {fmt['mcc']:.3f} / balanced accuracy {fmt['balanced_accuracy']:.3f} on this dataset")
    pure = ct[(ct["purity"] >= 0.99) & (ct["n"] >= 100)]
    if len(pure):
        flags.append(f"{len(pure)} format cell(s) with >=100 URLs are >=99% one class, covering {100 * pure['n'].sum() / len(y):.1f}% of the data")
    return AuditResult(name, len(y), class_table, ct, fmt, tpl, triv, nmi, flags)


def compare(train_urls: Sequence[str], train_labels: Sequence[int], test_urls: Sequence[str], test_labels: Sequence[int]) -> Dict[str, object]:
    """Transfer of the format-only rule from one dataset to another, and cell coverage of the target."""
    rule = FormatRule.fit(train_urls, train_labels)
    y = np.asarray(test_labels).astype(int)
    res = binary_metrics(y, rule.predict(test_urls))
    tr_cells = set(rule.cell_label)
    te = format_features(test_urls)
    return {"format_rule_transfer": res, "trivial_target": binary_metrics(y, np.ones(len(y), int)),
            "share_target_in_unseen_cells": float((~te["cell"].isin(tr_cells)).mean()),
            "train_cell_majority": rule.cell_label}
