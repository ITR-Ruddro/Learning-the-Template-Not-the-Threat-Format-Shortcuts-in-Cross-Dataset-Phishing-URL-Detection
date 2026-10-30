"""Revision analyses that need no retraining (reviewer points 1, 3, 4, 6 and minor points).
Outputs (results/):
  rev_A1_baselines_metrics.csv     trivial baselines, template rule and models: F1(phish), MCC, balanced accuracy,
                                   F1 and PR-AUC with LEGITIMATE as the positive class; seed mean + 95% URL-bootstrap CI + seed SD
  rev_A4_rule_agreement.csv        agreement and Cohen's kappa between each model (per seed) and the template rule
  rev_A4_disagreement_profile.csv  format profile of URLs on which model and rule disagree
  rev_A6_domain_overlap.csv        registered-domain overlap (leakage) between partitions and datasets
  rev_A6_domain_disjoint_metrics.csv  E1 metrics re-computed on registered-domain-disjoint subsets
  rev_A6_tld_profile.csv           top suffixes per set/class and char-CNN FPR per suffix group
  rev_A7_saturation.csv            logit saturation / ties (explains zero-width CIs and the DistilBERT LegitPhish oracle TPR)
  rev_A8_label_budget_threshold.csv threshold re-selected with a small labelled target sample (recalibration baseline)"""
import glob, re, numpy as np, pandas as pd
from common import *
from evaluate import weights_for, attach_groups

P = load_all()
TPL = re.compile(r'^https://www\.[a-z0-9.-]+$')
rng = np.random.default_rng(2026)


def load_orig(models):
    fs = [f'{ROOT}/predictions/lgbm_logits.csv.gz'] + sorted(glob.glob(f'{ROOT}/predictions/charcnn/charCNN_s*.csv.gz')) + \
         sorted(glob.glob(f'{ROOT}/predictions/transformer/DistilBERT_s*.csv.gz'))
    L = pd.concat([pd.read_csv(f) for f in fs]); L = L[(L.variant == 'orig') & L.model.isin(models)]
    return attach_groups(L, P)


L = load_orig(['LightGBM', 'charCNN', 'DistilBERT'])


def mets(pred, y, score, w):
    """pred: bool phishing; score: higher = more phishing. weighted metrics"""
    tp = np.sum(w * (pred & (y == 1))); fp = np.sum(w * (pred & (y == 0))); fn = np.sum(w * (~pred & (y == 1))); tn = np.sum(w * (~pred & (y == 0)))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1e-12)
    den = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-300))
    mcc = (tp * tn - fp * fn) / den if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0.0
    tpr = tp / max(tp + fn, 1e-12); tnr = tn / max(tn + fp, 1e-12)
    f1_leg = 2 * tn / max(2 * tn + fn + fp, 1e-12)
    ap_leg = wap(-score, (y == 0).astype(int), w) if score is not None else np.nan
    return {'F1_phish': f1, 'MCC': mcc, 'BalAcc': (tpr + tnr) / 2, 'F1_legit': f1_leg, 'PR_AUC_legit': ap_leg, 'TPR': tpr, 'TNR': tnr}


rows = []
for s in SETS:
    d0 = P[s].sort_values('row_id').reset_index(drop=True); y = d0.label.values; grp = d0.group.values
    W = weights_for(s, grp, y)
    rule = ~d0.url.str.strip().str.match(TPL).values
    for name, pred, score in [('all-phishing', np.ones(len(y), bool), None), ('all-legitimate', np.zeros(len(y), bool), None),
                              ('template rule', rule, None)]:
        pt = mets(pred, y, score, np.ones(len(y))); bs = [mets(pred, y, score, w.astype(float)) for w in W]
        r = {'model': name, 'set': s, 'n_seeds': 0}
        for k in pt:
            r[k] = pt[k]; v = [b[k] for b in bs]; r[k + '_lo'], r[k + '_hi'] = np.nanpercentile(v, [2.5, 97.5]); r[k + '_seed_sd'] = np.nan
        if score is None: r['PR_AUC_legit'] = r['PR_AUC_legit_lo'] = r['PR_AUC_legit_hi'] = (y == 0).mean() if name != 'template rule' else np.nan
        rows.append(r)
    for m in ['LightGBM', 'charCNN', 'DistilBERT']:
        pts, bts = [], []
        for sd in sorted(L[L.model == m].seed.unique()):
            x = L[(L.model == m) & (L.seed == sd) & (L.partition == s)].sort_values('row_id')
            z = x.logit.values; pred = z >= 0
            pts.append(mets(pred, y, z, np.ones(len(y)))); bts.append([mets(pred, y, z, w.astype(float)) for w in W])
        r = {'model': m, 'set': s, 'n_seeds': len(pts)}
        for k in pts[0]:
            r[k] = np.mean([p[k] for p in pts]); bm = np.mean([[b[k] for b in bl] for bl in bts], 0)
            r[k + '_lo'], r[k + '_hi'] = np.nanpercentile(bm, [2.5, 97.5]); r[k + '_seed_sd'] = np.std([p[k] for p in pts], ddof=1)
        rows.append(r)
    print('A1', s, flush=True)
A1 = pd.DataFrame(rows); A1.to_csv(f'{RES}/rev_A1_baselines_metrics.csv', index=False)


def kappa(a, b):
    po = np.mean(a == b); pa = a.mean(); pb = b.mean(); pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def fmt(u):
    sch, host, www, tail, slash = url_parts(u)
    return pd.Series({'https': sch == 'https', 'http': sch == 'http', 'www': www, 'trailing_slash': slash, 'path': len(tail) > 0})


ag, prof = [], []
for s in SETS:
    d0 = P[s].sort_values('row_id').reset_index(drop=True); y = d0.label.values
    rule = ~d0.url.str.strip().str.match(TPL).values
    F = d0.url.apply(fmt)
    for m in ['LightGBM', 'charCNN', 'DistilBERT']:
        for sd in sorted(L[L.model == m].seed.unique()):
            x = L[(L.model == m) & (L.seed == sd) & (L.partition == s)].sort_values('row_id'); pred = x.logit.values >= 0
            ag.append({'model': m, 'seed': sd, 'set': s, 'agreement': np.mean(pred == rule), 'kappa': kappa(pred, rule),
                       'agree_legit_class': np.mean((pred == rule)[y == 0]), 'agree_phish_class': np.mean((pred == rule)[y == 1]),
                       'n_model_phish_rule_legit': int((pred & ~rule).sum()), 'n_model_legit_rule_phish': int((~pred & rule).sum())})
            if sd == 42:
                for kind, msk in [('model phishing, rule legitimate', pred & ~rule), ('model legitimate, rule phishing', ~pred & rule), ('agree', pred == rule)]:
                    if msk.sum() == 0: continue
                    r = {'model': m, 'set': s, 'kind': kind, 'n': int(msk.sum()), 'pct_true_phishing': 100 * y[msk].mean()}
                    for c in F.columns: r['pct_' + c] = 100 * F[c].values[msk].mean()
                    prof.append(r)
    print('A4', s, flush=True)
A4 = pd.DataFrame(ag); A4.to_csv(f'{RES}/rev_A4_rule_agreement.csv', index=False)
pd.DataFrame(prof).to_csv(f'{RES}/rev_A4_disagreement_profile.csv', index=False)

# ---------------------------------------------------------------- registered domains (leakage / overlap / TLD)
def regdom(u):
    e = EXT(str(u).strip())
    return (e.top_domain_under_public_suffix if hasattr(e, 'top_domain_under_public_suffix') else e.registered_domain) or e.fqdn or 'NA', e.suffix or 'missing'


RD = {}
for k, d in P.items():
    rs = [regdom(u) for u in d.url.values]
    RD[k] = pd.DataFrame({'row_id': d.row_id.values, 'label': d.label.values, 'regdom': [r[0] for r in rs], 'suffix': [r[1] for r in rs]})
trv = pd.concat([RD['train'], RD['val']])
seen_all = set(trv.regdom); seen_leg = set(trv[trv.label == 0].regdom); seen_phi = set(trv[trv.label == 1].regdom)
ov = []
for k in SETS:
    d = RD[k]
    for cls, nm in [(0, 'legitimate'), (1, 'phishing')]:
        x = d[d.label == cls]
        ov.append({'set': k, 'class': nm, 'rows': len(x), 'distinct_regdomains': x.regdom.nunique(),
                   'pct_rows_regdomain_in_source_trainval': 100 * x.regdom.isin(seen_all).mean(),
                   'pct_rows_regdomain_in_source_trainval_legit': 100 * x.regdom.isin(seen_leg).mean(),
                   'pct_rows_regdomain_in_source_trainval_phish': 100 * x.regdom.isin(seen_phi).mean()})
pd.DataFrame(ov).to_csv(f'{RES}/rev_A6_domain_overlap.csv', index=False)

# metrics on registered-domain-disjoint subsets (rows whose registered domain never occurs in source train/val)
dd = []
for s in SETS:
    d = RD[s].sort_values('row_id').reset_index(drop=True); keep = ~d.regdom.isin(seen_all).values; y = d.label.values
    for m in ['LightGBM', 'charCNN', 'DistilBERT']:
        vals = []
        for sd in sorted(L[L.model == m].seed.unique()):
            x = L[(L.model == m) & (L.seed == sd) & (L.partition == s)].sort_values('row_id'); z = x.logit.values
            r = metrics(z[keep], y[keep]) if (y[keep] == 0).any() and (y[keep] == 1).any() else {}
            r.update(mets(z[keep] >= 0, y[keep], z[keep], np.ones(keep.sum()))); vals.append(r)
        out = {'model': m, 'set': s, 'subset': 'registered domain unseen in source train/val', 'n': int(keep.sum()),
               'n_legit': int((y[keep] == 0).sum()), 'n_phish': int((y[keep] == 1).sum())}
        for k in ['F1', 'PR_AUC', 'FPR', 'MCC', 'BalAcc']:
            if k in vals[0]: out[k] = np.mean([v[k] for v in vals])
        dd.append(out)
pd.DataFrame(dd).to_csv(f'{RES}/rev_A6_domain_disjoint_metrics.csv', index=False)

# suffix profile + char-CNN FPR by suffix group (URL-Phish legitimate)
tp = []
cn = L[(L.model == 'charCNN') & (L.partition == 'URL-Phish')].groupby('row_id').logit.mean()
for k in ['train', 'URL-Phish', 'LegitPhish']:
    d = RD[k]
    for cls, nm in [(0, 'legitimate'), (1, 'phishing')]:
        vc = d[d.label == cls].suffix.value_counts(normalize=True).head(8)
        for suf, f in vc.items(): tp.append({'set': k, 'class': nm, 'suffix': suf, 'pct_rows': 100 * f})
u = RD['URL-Phish']; u = u[u.label == 0].copy(); u['fp'] = (cn.reindex(u.row_id).values >= 0)
u['group'] = np.select([u.suffix.str.contains(r'(^|\.)(edu|ac)(\.|$)'), u.suffix.str.contains(r'(^|\.)(gov|gob|gouv|go)(\.|$)'),
                        u.suffix.eq('org'), u.suffix.eq('com')], ['edu/ac', 'gov', 'org', 'com'], 'other')
for g, x in u.groupby('group'):
    tp.append({'set': 'URL-Phish', 'class': 'legitimate', 'suffix': f'group:{g}', 'pct_rows': 100 * len(x) / len(u), 'charCNN_seedmean_FPR': x.fp.mean()})
pd.DataFrame(tp).to_csv(f'{RES}/rev_A6_tld_profile.csv', index=False)
print('A6 done', flush=True)

# ---------------------------------------------------------------- saturation / ties
sat = []
for s in SETS:
    for m in ['LightGBM', 'charCNN', 'DistilBERT']:
        for sd in sorted(L[L.model == m].seed.unique()):
            x = L[(L.model == m) & (L.seed == sd) & (L.partition == s)]; z = x.logit.values; y = x.label.values
            zl = z[y == 0]; thr = thr_at_fpr(zl, None, 0.01)
            sat.append({'model': m, 'seed': sd, 'set': s, 'distinct_logits': len(np.unique(np.round(z, 6))),
                        'pct_legit_logit_gt_5': 100 * np.mean(zl > 5), 'pct_phish_logit_gt_5': 100 * np.mean(z[y == 1] > 5),
                        'oracle_threshold_FPR1': thr, 'pct_phish_tied_at_or_above_threshold_tie': 100 * np.mean(np.isclose(z[y == 1], thr, atol=1e-6)),
                        'legit_logit_p99': np.percentile(zl, 99), 'phish_logit_median': np.median(z[y == 1])})
pd.DataFrame(sat).to_csv(f'{RES}/rev_A7_saturation.csv', index=False)

# ---------------------------------------------------------------- small-label threshold re-selection (recalibration baseline)
lb = []
_parts = {m: [L[L.model == m]] for m in ['LightGBM', 'charCNN', 'DistilBERT']}
for f in sorted(glob.glob(f'{ROOT}/predictions/charcnn/charCNN-*_s*.csv.gz')) + sorted(glob.glob(f'{ROOT}/predictions/transformer/DistilBERT-*_s*.csv.gz')):
    d = pd.read_csv(f); d = d[d.variant == 'orig']; _parts.setdefault(d.model.iloc[0], []).append(d)
mods = {m: pd.concat(v) for m, v in _parts.items()}
for m, D in mods.items():
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
            lb.append({'model': m, 'set': s, 'labelled_target_urls': budget, 'rule': 'max-F1 threshold on the labelled sample; evaluated on the rest',
                       'F1_mean': res[:, 0].mean(), 'F1_sd': res[:, 0].std(ddof=1), 'FPR_mean': res[:, 1].mean(), 'TPR_mean': res[:, 2].mean(), 'draws': len(res)})
pd.DataFrame(lb).to_csv(f'{RES}/rev_A8_label_budget_threshold.csv', index=False)
print('all done')
