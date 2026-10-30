"""Command-line interface: urlformat-audit {audit,compare,factorial}."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import __version__
from .audit import audit, compare
from .factorial import expand
from .io import binarize, read_table, resolve_columns


def _load(path, url_col, label_col, positive, extra=None):
    df = read_table(path)
    u, l = resolve_columns(df, url_col, label_col)
    urls = df[u].astype(str).tolist()
    if l is None:
        if extra is None:
            raise ValueError(f"{path}: no label column; pass --label-col (or --label for single-class files)")
        y = np.full(len(urls), int(extra))
    else:
        y = binarize(df[l], positive)
    keep = y >= 0
    if (~keep).any():
        print(f"note: {path}: {(~keep).sum()} row(s) without a label were skipped", file=sys.stderr)
    return [x for x, k in zip(urls, keep) if k], y[keep]


def _fmt(d):
    return "  ".join(f"{k}={v:.3f}" for k, v in d.items() if isinstance(v, float))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="urlformat-audit", description="Audit class-conditional URL formatting (shortcut risk) in labelled URL datasets.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--url-col"); p.add_argument("--label-col")
        p.add_argument("--positive", help="label value of the positive class (e.g. phishing, bad, 1)")

    a = sub.add_parser("audit", help="format audit of one labelled dataset")
    a.add_argument("path", nargs="+", help="one CSV/TSV/JSON file, or one file per class with --label")
    a.add_argument("--label", nargs="+", type=int, help="class (0/1) of each file when files have no label column")
    a.add_argument("--name", default=None); a.add_argument("--json", help="write the full result as JSON")
    a.add_argument("--cells", type=int, default=10, help="number of format cells to print")
    common(a)

    c = sub.add_parser("compare", help="transfer of the format-only rule from TRAIN to TEST")
    c.add_argument("train"); c.add_argument("test"); common(c)

    f = sub.add_parser("factorial", help="write the 24 factorial format variants of every URL")
    f.add_argument("path"); f.add_argument("-o", "--out", required=True); f.add_argument("--url-col")

    args = ap.parse_args(argv)
    if args.cmd == "audit":
        if len(args.path) > 1 or args.label:
            labs = args.label or []
            if len(labs) != len(args.path):
                ap.error("give one --label per file")
            parts = [_load(p, args.url_col, args.label_col, args.positive, extra=l) for p, l in zip(args.path, labs)]
            urls = sum([p[0] for p in parts], []); y = np.concatenate([p[1] for p in parts])
        else:
            urls, y = _load(args.path[0], args.url_col, args.label_col, args.positive)
        r = audit(urls, y, name=args.name or args.path[0])
        pd.set_option("display.width", 200)
        print(f"# {r.name}: {r.n:,} URLs ({100 * y.mean():.1f}% positive)\n")
        print("Format properties (% of URLs per class):")
        print(r.class_table.round(1).to_string(), "\n")
        print(f"Format cells (scheme|www|slash|path), top {args.cells}:")
        print(r.cell_table.head(args.cells).round(3).to_string(), "\n")
        print("Format-only rule (majority per cell):  ", _fmt(r.format_rule))
        print("Template rule (legit iff ^https://www.<host>$):", _fmt(r.template_rule))
        print("Trivial (all positive):                ", _fmt(r.trivial))
        print(f"Normalized mutual information I(cell;label)/H(label) = {r.nmi_cell_label:.3f}\n")
        print("Shortcut flags:" if r.flags else "No shortcut flags.")
        for fl in r.flags:
            print("  -", fl)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(r.to_dict(), fh, indent=1, default=float)
    elif args.cmd == "compare":
        tu, ty = _load(args.train, args.url_col, args.label_col, args.positive)
        su, sy = _load(args.test, args.url_col, args.label_col, args.positive)
        r = compare(tu, ty, su, sy)
        print("Format-only rule fitted on TRAIN, applied to TEST:", _fmt(r["format_rule_transfer"]))
        print("Trivial (all positive) on TEST:                   ", _fmt(r["trivial_target"]))
        print(f"Share of TEST URLs in format cells unseen in TRAIN: {100 * r['share_target_in_unseen_cells']:.1f}%")
    elif args.cmd == "factorial":
        df = read_table(args.path)
        u = args.url_col or next(c for c in ("url", "URL") if c in df.columns)
        out = expand(df[u].astype(str).tolist())
        out.to_csv(args.out, index=False)
        print(f"wrote {len(out):,} rows ({len(df):,} URLs x 24 cells) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
