"""Step 3 (E1 transfer + per-class errors) and Step 4 (E2 counterfactuals) from saved logits.
Usage: python step3_4_tables.py e1|e2|e4  [models...]
Logit sources: predictions/lgbm_logits.csv.gz, predictions/lgbm_variant_logits.csv.gz,
predictions/charcnn/*.csv.gz, predictions/transformer/*.csv.gz (uploaded from the GPU notebook)."""
import sys, glob, numpy as np, pandas as pd
from common import *
from evaluate import perf_table, cf_table, paired_delta, attach_groups

WHAT = sys.argv[1]
P = load_all()


def load(models=None, variants=None):
    fs = [f'{ROOT}/predictions/lgbm_logits.csv.gz', f'{ROOT}/predictions/lgbm_variant_logits.csv.gz'] + \
         sorted(glob.glob(f'{ROOT}/predictions/charcnn/*.csv.gz')) + sorted(glob.glob(f'{ROOT}/predictions/transformer/*.csv.gz'))
    out = []
    for f in fs:
        d = pd.read_csv(f)
        if 'changed' not in d: d['changed'] = d.variant != 'orig'
        if models: d = d[d.model.isin(models)]
        if variants: d = d[d.variant.isin(variants)]
        if len(d): out.append(d)
    L = pd.concat(out, ignore_index=True)
    return attach_groups(L, P)


if WHAT == 'e1':
    L = load(variants=['orig'])
    models = sys.argv[2:] or [m for m in ['LightGBM', 'charCNN', 'DistilBERT'] if m in set(L.model)]
    perf = pd.concat([perf_table(L, m) for m in models], ignore_index=True)
    perf.to_csv(f'{RES}/step3_E1_transfer.csv', index=False)
    # per-class error breakdown at threshold 0.5 (logit 0), seed-mean
    eb = []
    for m in models:
        for s in SETS:
            d = L[(L.model == m) & (L.partition == s)]
            g = d.assign(fp=(d.logit >= 0) & (d.label == 0), fn=(d.logit < 0) & (d.label == 1)).groupby('seed')
            fp, fn = g.fp.sum().mean(), g.fn.sum().mean()
            nl, nph = (d[d.seed == d.seed.iloc[0]].label == 0).sum(), (d[d.seed == d.seed.iloc[0]].label == 1).sum()
            eb.append({'model': m, 'set': s, 'n_legit': int(nl), 'n_phish': int(nph), 'FP_mean': fp, 'FN_mean': fn,
                       'FPR': fp / nl, 'FNR': fn / nph, 'share_of_errors_FP': fp / max(fp + fn, 1e-12)})
    pd.DataFrame(eb).to_csv(f'{RES}/step3_E1_error_breakdown.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(perf[['model', 'set', 'F1', 'F1_lo', 'F1_hi', 'PR_AUC', 'PR_AUC_lo', 'PR_AUC_hi', 'FPR', 'FPR_lo', 'FPR_hi', 'TPR_at_valFPR1', 'achFPR_valthr',
                'TPR_at_oracleFPR1', 'achFPR_oraclethr', 'Brier', 'ECE15']].round(4).to_string(index=False))
    print(pd.DataFrame(eb).round(4).to_string(index=False))

if WHAT == 'e2':
    import gc
    models = sys.argv[2:] or ['LightGBM', 'charCNN', 'DistilBERT']
    res = []
    sch = pd.concat([P[k][['row_id']].assign(partition=k, orig_scheme=[(SCH.match(u.strip()).group(1).lower() if SCH.match(u.strip()) else 'none') for u in P[k].url]) for k in SETS])
    for m in models:  # one model at a time (memory)
        L = load(models=[m]); L = L[L.partition != 'val'].merge(sch, on=['row_id', 'partition'], how='left')
        L['changed'] = L.changed.astype(bool)
        for v in sorted(set(L.variant) - {'orig'}):
            t = cf_table(L, m, v); t['direction'] = 'all'; res.append(t)
            if v in ('swap', 'IsHTTPS_flip'):  # split by direction of the scheme edit
                keep = L.changed.values.copy()
                for o_s, nm in [('https', 'https->http'), ('http', 'http->https')]:
                    L['changed'] = keep & (L.orig_scheme.values == o_s)
                    t = cf_table(L, m, v); t['direction'] = nm; res.append(t)
                L['changed'] = keep
            print(m, v, flush=True)
        del L; gc.collect()
    res = pd.concat(res, ignore_index=True)
    res.to_csv(f'{RES}/step4_E2_counterfactual.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(res[['model', 'variant', 'direction', 'set', 'cls', 'n_changed', 'orig_error_rate', 'mean_dlogit', 'mean_dlogit_lo', 'mean_dlogit_hi',
               'flip_rate', 'pct_errors_corrected', 'pct_correct_broken']].round(4).to_string(index=False))

if WHAT == 'e4':
    L = load(variants=['orig'])
    fam = sys.argv[2] if len(sys.argv) > 2 else 'charCNN'
    rows, perf = [], []
    for suffix in ['noscheme', 'swapaug', 'canonical']:
        b = f'{fam}-{suffix}'
        if b not in set(L.model): continue
        rows.append(paired_delta(L, fam, 'orig', b, 'orig', keys=('F1', 'PR_AUC', 'FPR', 'FNR')))
        perf.append(perf_table(L, b))
    pd.concat(rows, ignore_index=True).to_csv(f'{RES}/step6_E4_{fam}_paired_delta.csv', index=False)
    pd.concat(perf, ignore_index=True).to_csv(f'{RES}/step6_E4_{fam}_performance.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(pd.concat(rows).round(4).to_string(index=False))
