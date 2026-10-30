"""Evaluation of saved per-row logits: performance metrics (E1/E4) and counterfactual deltas (E2).
All CIs: 1,000 class-stratified exact-URL-group bootstrap resamples; the same resamples (fixed seed per set)
are used for every model/variant so differences are paired. Seed-mean is bootstrapped (metric per seed, averaged)."""
import numpy as np, pandas as pd
from common import *

BOOT_SEED = {'PhiUSIIL test': 1001, 'URL-Phish': 1002, 'LegitPhish': 1003}
_WCACHE = {}


def weights_for(setname, group, y, B=1000):
    key = (setname, len(y), B)
    if key not in _WCACHE:
        _WCACHE[key] = np.array(list(boot_weights(group, y, B, BOOT_SEED[setname])), dtype=np.float32)
    return _WCACHE[key]


def val_threshold(zval, yval):
    return thr_at_fpr(zval[yval == 0], None, 0.01)


def perf_table(L, model, variant='orig', B=1000):
    """L: logits frame with columns row_id, partition, label, logit, seed, model, variant, group.
    Returns one row per evaluated set with seed-mean metrics and 95% CIs."""
    out = []
    d = L[(L.model == model) & (L.variant == variant)]
    seeds = sorted(d.seed.unique())
    for s in SETS:
        pts, bts = [], []
        for sd in seeds:
            v = L[(L.model == model) & (L.variant == 'orig') & (L.seed == sd) & (L.partition == 'val')]
            if variant != 'orig' and len(L[(L.model == model) & (L.variant == variant) & (L.seed == sd) & (L.partition == 'val')]):
                v = L[(L.model == model) & (L.variant == variant) & (L.seed == sd) & (L.partition == 'val')]
            t = val_threshold(v.logit.values, v.label.values)
            x = d[(d.seed == sd) & (d.partition == s)].sort_values('row_id')
            z, y = x.logit.values, x.label.values
            W = weights_for(s, x.group.values, y, B)
            order = np.argsort(-z, kind='mergesort')
            pts.append(metrics(z, y, None, t, order))
            bts.append([metrics(z, y, w.astype(float), t, order) for w in W])
        r = summarize(pts, bts); r.update(model=model, variant=variant, set=s, n=len(x), n_seeds=len(seeds))
        out.append(r)
    return pd.DataFrame(out)


def paired_delta(L, model_a, var_a, model_b, var_b, keys=('F1', 'PR_AUC', 'FPR'), B=1000):
    """Delta (b - a) of seed-mean metrics with paired bootstrap CI (same resamples)."""
    out = []
    common = sorted(set(L[(L.model == model_a) & (L.variant == var_a)].seed) & set(L[(L.model == model_b) & (L.variant == var_b)].seed))
    for s in SETS:
        db, da = {}, {}
        for tag, (m, v) in {'a': (model_a, var_a), 'b': (model_b, var_b)}.items():
            d = L[(L.model == m) & (L.variant == v)]
            pts, bts = [], []
            for sd in common:
                x = d[(d.seed == sd) & (d.partition == s)].sort_values('row_id')
                z, y = x.logit.values, x.label.values
                W = weights_for(s, x.group.values, y, B); order = np.argsort(-z, kind='mergesort')
                pts.append(metrics(z, y, None, None, order)); bts.append([metrics(z, y, w.astype(float), None, order) for w in W])
            (da if tag == 'a' else db).update(pt=pts, bt=bts)
        r = {'set': s, 'a': f'{model_a}/{var_a}', 'b': f'{model_b}/{var_b}', 'seeds': ' '.join(map(str, common))}
        for k in keys:
            A = np.mean([p[k] for p in da['pt']]); Bv = np.mean([p[k] for p in db['pt']])
            ba = np.mean([[q[k] for q in bl] for bl in da['bt']], 0); bb = np.mean([[q[k] for q in bl] for bl in db['bt']], 0)
            dd = bb - ba
            r.update({f'{k}_a': A, f'{k}_b': Bv, f'd{k}': Bv - A, f'd{k}_lo': np.percentile(dd, 2.5), f'd{k}_hi': np.percentile(dd, 97.5)})
        out.append(r)
    return pd.DataFrame(out)


def cf_table(L, model, variant, B=1000):
    """Counterfactual: per set x class, mean delta-logit (variant - orig) over rows where the edit changes the string,
    flip rate (decision at logit 0 changes), and share of original errors corrected. Seed-mean with group-bootstrap CI."""
    out = []
    o = L[(L.model == model) & (L.variant == 'orig')]; c = L[(L.model == model) & (L.variant == variant)]
    seeds = sorted(c.seed.unique())
    for s in SETS:
        base = {sd: o[(o.seed == sd) & (o.partition == s)].sort_values('row_id') for sd in seeds}
        cfd = {sd: c[(c.seed == sd) & (c.partition == s)].sort_values('row_id') for sd in seeds}
        x0 = base[seeds[0]]; y = x0.label.values; grp = x0.group.values
        changed = cfd[seeds[0]].changed.values.astype(bool)
        W = weights_for(s, grp, y, B)
        for cls, nm in [(0, 'legitimate'), (1, 'phishing')]:
            m = (y == cls) & changed
            if m.sum() == 0:
                out.append(dict(model=model, variant=variant, set=s, cls=nm, n_class=int((y == cls).sum()), n_changed=0)); continue
            dz = np.stack([cfd[sd].logit.values - base[sd].logit.values for sd in seeds])  # seeds x rows
            p0 = np.stack([base[sd].logit.values >= 0 for sd in seeds]); p1 = np.stack([cfd[sd].logit.values >= 0 for sd in seeds])
            flip = (p0 != p1)
            err0 = p0 != (y == 1); corrected = err0 & ~(p1 != (y == 1)); broken = ~err0 & (p1 != (y == 1))

            def stats(w):
                ww = w * m
                tot = ww.sum()
                e = (err0 * ww).sum(1)
                return [(dz * ww).sum(1).mean() / tot, (flip * ww).sum(1).mean() / tot,
                        np.mean((corrected * ww).sum(1) / np.maximum(e, 1e-12)), np.mean((broken * ww).sum(1) / tot)]
            pt = stats(np.ones(len(y))); bs = np.array([stats(w.astype(float)) for w in W])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            r = dict(model=model, variant=variant, set=s, cls=nm, n_class=int((y == cls).sum()), n_changed=int(m.sum()),
                     orig_error_rate=float((err0 * m).sum(1).mean() / m.sum()))
            for i, k in enumerate(['mean_dlogit', 'flip_rate', 'pct_errors_corrected', 'pct_correct_broken']):
                r[k], r[k + '_lo'], r[k + '_hi'] = float(pt[i]), float(lo[i]), float(hi[i])
            r['mean_dlogit_seed_sd'] = float(np.std([dz[i][m].mean() for i in range(len(seeds))], ddof=1)) if len(seeds) > 1 else np.nan
            out.append(r)
    return pd.DataFrame(out)


def attach_groups(L, P):
    g = pd.concat([P[k][['row_id', 'group']].assign(partition=k) for k in ['val'] + SETS])
    return L.merge(g, on=['row_id', 'partition'], how='left')
