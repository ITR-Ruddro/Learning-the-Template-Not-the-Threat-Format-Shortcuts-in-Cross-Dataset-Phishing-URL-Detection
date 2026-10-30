"""Revision D (reviewer point 9 + threshold remarks): E4 recomputed with the char-CNN variants re-trained under a 20-epoch cap
(predictions/charcnn_long/), plus the new host-only variant. DistilBERT variants as available in predictions/transformer/.
Outputs: results/rev_D_E4_paired_delta.csv, results/rev_D_E4_performance.csv (incl. TPR@val, MCC via rev_A1-style metrics),
         results/rev_D_E4_label_budget.csv (threshold re-selected on 200 / 1,000 labelled target URLs), results/rev_D_E4_fpr_by_path.csv"""
import glob, numpy as np, pandas as pd
from common import *
from evaluate import perf_table, paired_delta, attach_groups

P = load_all()
fs = sorted(glob.glob(f'{ROOT}/predictions/charcnn/charCNN_s*.csv.gz')) + sorted(glob.glob(f'{ROOT}/predictions/charcnn_long/*.csv.gz')) + \
     sorted(glob.glob(f'{ROOT}/predictions/transformer/DistilBERT*_s*.csv.gz'))
def _rd(f):
    d = pd.read_csv(f, usecols=['row_id', 'partition', 'label', 'logit', 'variant', 'seed', 'model'])
    return d[d.variant == 'orig']
L = attach_groups(pd.concat([_rd(f) for f in fs], ignore_index=True), P)
VARS = ['noscheme', 'swapaug', 'canonical', 'hostonly']
dl, pf = [], []
for fam in ['charCNN', 'DistilBERT']:
    pf.append(perf_table(L, fam))
    for v in VARS:
        b = f'{fam}-{v}'
        if b not in set(L.model): continue
        dl.append(paired_delta(L, fam, 'orig', b, 'orig', keys=('F1', 'PR_AUC', 'FPR', 'FNR')))
        pf.append(perf_table(L, b)); print(b, flush=True)
pd.concat(dl, ignore_index=True).to_csv(f'{RES}/rev_D_E4_paired_delta.csv', index=False)
PF = pd.concat(pf, ignore_index=True)
# MCC (seed mean) at p = 0.5
mc = []
for (m, s), x in L[L.partition.isin(SETS)].groupby(['model', 'partition']):
    v = []
    for sd, g in x.groupby('seed'):
        y = g.label.values; pr = g.logit.values >= 0
        tp = float(np.sum(pr & (y == 1))); fp = float(np.sum(pr & (y == 0))); fn = float(np.sum(~pr & (y == 1))); tn = float(np.sum(~pr & (y == 0)))
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); v.append((tp * tn - fp * fn) / den if den > 0 else 0.0)
    mc.append({'model': m, 'set': s, 'MCC': np.mean(v), 'MCC_seed_sd': np.std(v, ddof=1) if len(v) > 1 else np.nan, 'n_seeds_mcc': len(v)})
PF = PF.merge(pd.DataFrame(mc), on=['model', 'set'], how='left'); PF.to_csv(f'{RES}/rev_D_E4_performance.csv', index=False)

# small-label threshold re-selection
lb = []
for m, D in L[L.partition.isin(['URL-Phish', 'LegitPhish'])].groupby('model'):
    for s in ['URL-Phish', 'LegitPhish']:
        for budget in [200, 1000]:
            res = []
            for sd in sorted(D.seed.unique()):
                x = D[(D.seed == sd) & (D.partition == s)].sort_values('row_id'); z = x.logit.values; y = x.label.values
                for rep in range(20):
                    r2 = np.random.default_rng(1000 * sd + rep)
                    idx = np.concatenate([r2.choice(np.where(y == c)[0], int(round(budget * (y == c).mean())), replace=False) for c in (0, 1)])
                    mask = np.ones(len(y), bool); mask[idx] = False
                    zs, ys = z[idx], y[idx]; cands = np.unique(zs)
                    f1s = [2 * np.sum((zs >= t) & (ys == 1)) / max(np.sum(zs >= t) + np.sum(ys == 1), 1) for t in cands]
                    t = cands[int(np.argmax(f1s))]; pr = z[mask] >= t; yy = y[mask]
                    tpv = np.sum(pr & (yy == 1)); fpv = np.sum(pr & (yy == 0)); fnv = np.sum(~pr & (yy == 1))
                    res.append((2 * tpv / max(2 * tpv + fpv + fnv, 1), fpv / np.sum(yy == 0), tpv / np.sum(yy == 1)))
            res = np.array(res)
            lb.append({'model': m, 'set': s, 'labelled_target_urls': budget, 'F1_mean': res[:, 0].mean(), 'F1_sd': res[:, 0].std(ddof=1),
                       'FPR_mean': res[:, 1].mean(), 'TPR_mean': res[:, 2].mean(), 'draws': len(res), 'n_seeds': D.seed.nunique()})
pd.DataFrame(lb).to_csv(f'{RES}/rev_D_E4_label_budget.csv', index=False)

# FPR by path (URL-Phish legitimate)
hp = {r: len(url_parts(u)[3]) > 0 for r, u in zip(P['URL-Phish'].row_id, P['URL-Phish'].url)}
x = L[(L.partition == 'URL-Phish') & (L.label == 0)].copy(); x['has_path'] = x.row_id.map(hp)
x['fp'] = x.logit >= 0
x.groupby(['model', 'has_path']).fp.mean().reset_index().rename(columns={'fp': 'FPR_seedpooled'}).to_csv(f'{RES}/rev_D_E4_fpr_by_path.csv', index=False)
pd.set_option('display.width', 250)
print(pd.read_csv(f'{RES}/rev_D_E4_paired_delta.csv')[['b', 'seeds', 'set', 'F1_a', 'F1_b', 'dF1', 'dF1_lo', 'dF1_hi', 'PR_AUC_b', 'FPR_b', 'FNR_b']].round(3).to_string(index=False))
print(PF[PF.set == 'URL-Phish'][['model', 'F1', 'PR_AUC', 'FPR', 'TPR_at_valFPR1', 'achFPR_valthr', 'MCC']].round(3).to_string(index=False))
print(pd.read_csv(f'{RES}/rev_D_E4_label_budget.csv').round(3).to_string(index=False))
print(pd.read_csv(f'{RES}/rev_D_E4_fpr_by_path.csv').round(3).to_string(index=False))
