"""Revision C (reviewer points 7 and 10).
C1  LightGBM+format: the 15 features plus has_www, trailing_slash, has_path, path_length (PhiUSIIL -> targets, seeds 42-44).
C2  Reverse transfer: train on URL-Phish (URL-grouped split, same procedure as for PhiUSIIL) -> PhiUSIIL and LegitPhish.
C3  Mixed source: train on PhiUSIIL-train + URL-Phish-train -> each test partition and LegitPhish.
Models: LightGBM, LightGBM+format, char-CNN (seeds 42-44). Writes predictions/rev_c/*.csv.gz and results/rev_C_transfer.csv
(point metrics, seed mean and SD; thresholds at p = 0.5 and FPR<=1% chosen on the training source's validation split)."""
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier
sys.argv = [sys.argv[0], 'orig', '42']
from step2_charcnn import CharCNN, encode, batchify, batches_by_length
from common import *

torch.set_num_threads(2)
WHICH = os.environ.get('REV_C', 'C1,C2,C3').split(',')
t0 = time.time(); P = load_all(); OUT = f'{ROOT}/predictions/rev_c'; os.makedirs(OUT, exist_ok=True)
FMT = ['has_www', 'trailing_slash', 'has_path', 'path_length']


def fmt_feats(urls):
    out = []
    for u in urls:
        sch, host, www, tail, slash = url_parts(u)
        out.append((int(www), int(slash), int(len(tail) > 0), len(tail)))
    return pd.DataFrame(out, columns=FMT)


def featurize(df):
    f = feature_frame(df.url.values)
    return pd.concat([f, fmt_feats(df.url.values)], axis=1)


def lgbm(seed):
    return LGBMClassifier(n_estimators=100, learning_rate=.05, max_depth=5, num_leaves=31, subsample=.8, subsample_freq=0, colsample_bytree=.8,
                          class_weight='balanced', random_state=seed, n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)


def run_lgbm(train, evals, cols, seed):
    enc = ColumnTransformer([('n', 'passthrough', [c for c in cols if c != 'TLD']),
                             ('t', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64), ['TLD'])], sparse_threshold=1.0)
    enc.fit(train['F'][cols]); m = lgbm(seed).fit(sparse.csr_matrix(enc.transform(train['F'][cols])), train['df'].label.values)
    return {k: m.booster_.predict(sparse.csr_matrix(enc.transform(v['F'][cols])), raw_score=True) for k, v in evals.items()}


@torch.no_grad()
def cnn_predict(model, urls, bs=2048):
    model.eval(); seqs = encode([u.strip() for u in urls]); out = np.zeros(len(seqs), np.float32)
    for ch in batches_by_length(seqs, bs, False):
        x, ln = batchify([seqs[i] for i in ch]); out[ch] = model(x, ln).numpy()
    return out


def run_cnn(train, val, evals, seed, max_epochs=20):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    model = CharCNN(); opt = torch.optim.Adam(model.parameters(), 1e-3); lossf = nn.BCEWithLogitsLoss()
    seqs = encode([u.strip() for u in train.url.values]); ytr = train.label.values.astype(np.float32); yva = val.label.values
    best, state, log = np.inf, None, []
    for ep in range(1, max_epochs + 1):
        model.train()
        for ch in batches_by_length(seqs, 256, True, rng):
            ch = rng.permutation(ch); x, ln = batchify([seqs[i] for i in ch])
            opt.zero_grad(); loss = lossf(model(x, ln), torch.from_numpy(ytr[ch])); loss.backward(); opt.step()
        zv = cnn_predict(model, val.url.values); vl = float(np.mean(np.logaddexp(0, zv) - yva * zv)); log.append(vl)
        if vl < best - 1e-4: best, state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        else: break
    model.load_state_dict(state)
    return {k: cnn_predict(model, v['df'].url.values) for k, v in evals.items()}, len(log)


def grouped_split(df, seed=42):
    g = pd.factorize(df.url.str.strip())[0]; y = df.label.values
    dev, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(df)), y, g))
    tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(dev)), y[dev], g[dev]))
    return np.sort(dev[tr]), np.sort(dev[va]), np.sort(te)


def evaluate(name, preds, sets, val_key, rows, extra):
    for k, d in sets.items():
        if k == val_key: continue
        for sd, z in preds.items():
            zv = preds[sd][val_key]; yv = sets[val_key]['df'].label.values; t = thr_at_fpr(zv[yv == 0], None, 0.01)
            y = d['df'].label.values; r = metrics(z[k], y, None, t)
            tp = np.sum((z[k] >= 0) & (y == 1)); fp = np.sum((z[k] >= 0) & (y == 0)); fn = np.sum((z[k] < 0) & (y == 1)); tn = np.sum((z[k] < 0) & (y == 0))
            den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
            r['MCC'] = (tp * tn - fp * fn) / den if den > 0 else 0.0; r['BalAcc'] = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
            r['trivial_allphish_F1'] = 2 * y.mean() / (1 + y.mean())
            rows.append({'experiment': extra, 'model': name, 'seed': sd, 'eval_set': k, 'n': len(y), 'phish_rate': y.mean(), **r})


rows = []
# ------------------------------------------------------------------ C1: format features, PhiUSIIL source
if 'C1' in WHICH:
    S = {k: {'df': P[k], 'F': featurize(P[k])} for k in ['train', 'val'] + SETS}
    for name, cols in [('LightGBM', FEATURES), ('LightGBM+format', FEATURES + FMT)]:
        preds = {sd: run_lgbm(S['train'], {k: S[k] for k in ['val'] + SETS}, cols, sd) for sd in SEEDS}
        evaluate(name, preds, {k: S[k] for k in ['val'] + SETS}, 'val', rows, 'C1 PhiUSIIL source')
        # template projection (path kept) and host-only template on URL-Phish legitimate false positives
        if name == 'LightGBM+format':
            for sd in SEEDS:
                pd.DataFrame({'row_id': P['URL-Phish'].row_id, 'label': P['URL-Phish'].label, 'logit': preds[sd]['URL-Phish'], 'seed': sd,
                              'model': name}).to_csv(f'{OUT}/C1_{name}_s{sd}.csv.gz', index=False, float_format='%.6f')
    print('C1', round(time.time() - t0), flush=True)

# ------------------------------------------------------------------ C2: reverse transfer, URL-Phish source
if 'C2' in WHICH or 'C3' in WHICH:
    up = pd.read_csv(f'{DATA}/urlphish.csv' if os.path.exists(f'{DATA}/urlphish.csv') else f'{DATA}/URL-Phish.csv', usecols=['url', 'label'])
    up = pd.DataFrame({'row_id': [f'URL-Phish:{i}' for i in range(len(up))], 'dataset': 'URL-Phish', 'url': up.url.astype(str), 'label': up.label.astype(int)})
    utr, uva, ute = grouped_split(up)
    ph = pd.concat([P['train'], P['val'], P['PhiUSIIL test']]).sort_values('row_id', key=lambda s: s.str.split(':').str[1].astype(int)).reset_index(drop=True)
    lp = P['LegitPhish']
if 'C2' in WHICH:
    seen = set(up.url.iloc[np.r_[utr, uva]].str.strip())
    T = {'train': up.iloc[utr].reset_index(drop=True), 'val': up.iloc[uva].reset_index(drop=True), 'URL-Phish test': up.iloc[ute].reset_index(drop=True),
         'PhiUSIIL (all, unseen URLs)': ph[~ph.url.str.strip().isin(seen)].reset_index(drop=True), 'LegitPhish (unseen URLs)': lp[~lp.url.str.strip().isin(seen)].reset_index(drop=True)}
    S = {k: {'df': v, 'F': featurize(v)} for k, v in T.items()}
    for name, cols in [('LightGBM', FEATURES), ('LightGBM+format', FEATURES + FMT)]:
        preds = {sd: run_lgbm(S['train'], {k: S[k] for k in S if k != 'train'}, cols, sd) for sd in SEEDS}
        evaluate(name, preds, {k: S[k] for k in S if k != 'train'}, 'val', rows, 'C2 URL-Phish source')
    preds = {}
    for sd in SEEDS:
        preds[sd], ne = run_cnn(T['train'], T['val'], {k: S[k] for k in S if k != 'train'}, sd); print('C2 cnn', sd, ne, round(time.time() - t0), flush=True)
    evaluate('charCNN', preds, {k: S[k] for k in S if k != 'train'}, 'val', rows, 'C2 URL-Phish source')
    pd.DataFrame(rows).to_csv(f'{RES}/rev_C_transfer.csv', index=False)

# ------------------------------------------------------------------ C3: mixed source
if 'C3' in WHICH:
    tr = pd.concat([P['train'], up.iloc[utr]]).reset_index(drop=True); va = pd.concat([P['val'], up.iloc[uva]]).reset_index(drop=True)
    seen = set(pd.concat([tr, va]).url.str.strip())
    T = {'train': tr, 'val': va, 'PhiUSIIL test': P['PhiUSIIL test'], 'URL-Phish test': up.iloc[ute].reset_index(drop=True),
         'LegitPhish (unseen URLs)': lp[~lp.url.str.strip().isin(seen)].reset_index(drop=True)}
    S = {k: {'df': v, 'F': featurize(v)} for k, v in T.items()}
    for name, cols in [('LightGBM', FEATURES), ('LightGBM+format', FEATURES + FMT)]:
        preds = {sd: run_lgbm(S['train'], {k: S[k] for k in S if k != 'train'}, cols, sd) for sd in SEEDS}
        evaluate(name, preds, {k: S[k] for k in S if k != 'train'}, 'val', rows, 'C3 mixed source')
    preds = {}
    for sd in SEEDS:
        preds[sd], ne = run_cnn(T['train'], T['val'], {k: S[k] for k in S if k != 'train'}, sd); print('C3 cnn', sd, ne, round(time.time() - t0), flush=True)
    evaluate('charCNN', preds, {k: S[k] for k in S if k != 'train'}, 'val', rows, 'C3 mixed source')

R = pd.DataFrame(rows); R.to_csv(f'{RES}/rev_C_transfer.csv', index=False)
agg = R.groupby(['experiment', 'model', 'eval_set']).agg(n=('n', 'first'), phish_rate=('phish_rate', 'first'), trivial_F1=('trivial_allphish_F1', 'first'),
                                                         F1=('F1', 'mean'), F1_sd=('F1', 'std'), PR_AUC=('PR_AUC', 'mean'), FPR=('FPR', 'mean'), FNR=('FNR', 'mean'),
                                                         MCC=('MCC', 'mean'), BalAcc=('BalAcc', 'mean'), TPR_valthr=('TPR_at_valFPR1', 'mean'), achFPR_valthr=('achFPR_valthr', 'mean')).reset_index()
agg.to_csv(f'{RES}/rev_C_transfer_summary.csv', index=False)
pd.set_option('display.width', 250); print(agg.round(3).to_string(index=False)); print('done', round(time.time() - t0))
