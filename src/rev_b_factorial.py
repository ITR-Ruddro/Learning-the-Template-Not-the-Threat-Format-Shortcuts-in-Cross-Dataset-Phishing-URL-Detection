"""Revision B (reviewer points 2 and 5): full factorial format edits scheme(https/http/none) x www(yes/no) x trailing slash(yes/no)
x path(kept/removed) = 24 cells, applied to every evaluated URL, for LightGBM and the unmodified char-CNN (seeds 42-44).
Outputs (results/):
  rev_B1_factorial_cells.csv      per model/seed-mean, set, class, cell: share predicted phishing, mean logit, training support of the cell
  rev_B2_factorial_anova.csv      share of between-cell variance of the mean logit explained by main effects vs. interactions
  rev_B3_residual_fp_by_path.csv  URL-Phish false positives corrected by the path-kept vs. host-only template, split by whether the URL has a path
  rev_B4_canonical_residual_by_path.csv  remaining false positives of canonical-trained models, split by path
  predictions/factorial/<model>_s<seed>.npz  per-row logits for all 24 cells"""
import sys, time, glob, numpy as np, pandas as pd, torch
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
sys.argv = [sys.argv[0], 'orig', '42']
from step2_charcnn import CharCNN, encode, batchify, batches_by_length
from common import *

torch.set_num_threads(2)
t0 = time.time(); P = load_all(); os.makedirs(f'{ROOT}/predictions/factorial', exist_ok=True)
CELLNAME = [f'{sc}|{"www" if w else "nowww"}|{"slash" if sl else "noslash"}|{"path" if p else "nopath"}' for sc, w, sl, p in FACT_CELLS]
STR = {}  # set -> list of 24 arrays of strings
for s in SETS:
    urls = P[s].url.values
    STR[s] = [np.array([factorial_variant(u, *c) for u in urls], dtype=object) for c in FACT_CELLS]
print('strings built', round(time.time() - t0), flush=True)

# training support of each cell (format of the real source-training URLs)
tr = pd.DataFrame([url_parts(u) for u in P['train'].url.values], columns=['scheme', 'host', 'www', 'tail', 'slash'])
tr['scheme'] = tr.scheme.fillna('none'); tr['path'] = tr['tail'].str.len() > 0; tr['label'] = P['train'].label.values
support = {}
for cn, (sc, w, sl, p) in zip(CELLNAME, FACT_CELLS):
    m = (tr.scheme == sc) & (tr.www == w) & (tr.slash == sl) & (tr.path == p)
    support[cn] = (int((m & (tr.label == 0)).sum()), int((m & (tr.label == 1)).sum()))

# ---------------------------------------------------------------- LightGBM (refit, deterministic; identical to step 1)
F = {k: pd.read_pickle(f"{RES}/cache/feat_{k.replace(' ', '_')}_orig.pkl") for k in P}
enc = ColumnTransformer([('numeric', StandardScaler(), NUMERIC), ('tld', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64), ['TLD'])], sparse_threshold=1.0)
enc.fit(F['train'][FEATURES]); Xtr = sparse.csr_matrix(enc.transform(F['train'][FEATURES]))
XF = None if all(os.path.exists(f'{ROOT}/predictions/factorial/LightGBM_s{sd}.npz') for sd in SEEDS) else \
     {s: [sparse.csr_matrix(enc.transform(feature_frame(a)[FEATURES])) for a in STR[s]] for s in SETS}
print('lgbm features', round(time.time() - t0), flush=True)
LOG = {}
def _load(m, sd):
    f = f'{ROOT}/predictions/factorial/{m}_s{sd}.npz'
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=True)
    d = {s: z[f'{s}|cells'] for s in SETS}; d['orig'] = {s: z[f'{s}|orig'] for s in SETS}; return d
def _save(m, sd, d):
    np.savez_compressed(f'{ROOT}/predictions/factorial/{m}_s{sd}.npz', **{f'{s}|cells': d[s].astype(np.float32) for s in SETS},
                        **{f'{s}|orig': d['orig'][s].astype(np.float32) for s in SETS}, cells=np.array(CELLNAME))
for sd in SEEDS:
    if _load('LightGBM', sd) is not None: LOG['LightGBM', sd] = _load('LightGBM', sd); continue
    m = LGBMClassifier(n_estimators=100, learning_rate=.05, max_depth=5, num_leaves=31, subsample=.8, subsample_freq=0, colsample_bytree=.8,
                       class_weight='balanced', random_state=sd, n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True).fit(Xtr, P['train'].label.values)
    LOG['LightGBM', sd] = {s: np.stack([m.booster_.predict(x, raw_score=True) for x in XF[s]]) for s in SETS}
    LOG['LightGBM', sd]['orig'] = {s: m.booster_.predict(sparse.csr_matrix(enc.transform(F[s][FEATURES])), raw_score=True) for s in SETS}
    _save('LightGBM', sd, LOG['LightGBM', sd])


# ---------------------------------------------------------------- char-CNN (saved models)
@torch.no_grad()
def cnn_predict(model, urls, bs=2048):
    seqs = encode([u.strip() for u in urls]); out = np.zeros(len(seqs), np.float32)
    for ch in batches_by_length(seqs, bs, False):
        x, ln = batchify([seqs[i] for i in ch]); out[ch] = model(x, ln).numpy()
    return out


for sd in SEEDS:
    if _load('charCNN', sd) is not None: LOG['charCNN', sd] = _load('charCNN', sd); continue
    net = CharCNN(); net.load_state_dict(torch.load(f'{ROOT}/models/charCNN_s{sd}.pt')); net.eval()
    LOG['charCNN', sd] = {s: np.stack([cnn_predict(net, a) for a in STR[s]]) for s in SETS}
    LOG['charCNN', sd]['orig'] = {s: cnn_predict(net, P[s].url.values) for s in SETS}
    _save('charCNN', sd, LOG['charCNN', sd]); print('charCNN', sd, round(time.time() - t0), flush=True)

# ---------------------------------------------------------------- B1 cell table
rows = []
for m in ['LightGBM', 'charCNN']:
    for s in SETS:
        y = P[s].label.values
        for ci_, cn in enumerate(CELLNAME):
            for cls, nm in [(0, 'legitimate'), (1, 'phishing')]:
                k = y == cls
                ph = np.mean([np.mean(LOG[m, sd][s][ci_][k] >= 0) for sd in SEEDS])
                lg = np.mean([np.mean(LOG[m, sd][s][ci_][k]) for sd in SEEDS])
                sc, w, sl, p = cn.split('|')
                rows.append({'model': m, 'set': s, 'class': nm, 'cell': cn, 'scheme': sc, 'www': w, 'slash': sl, 'path': p,
                             'share_pred_phishing': ph, 'mean_logit': lg,
                             'is_source_template': cn == 'https|www|noslash|nopath',
                             'train_support_legit': support[cn][0], 'train_support_phish': support[cn][1]})
B1 = pd.DataFrame(rows); B1.to_csv(f'{RES}/rev_B1_factorial_cells.csv', index=False)

# ---------------------------------------------------------------- B2 variance decomposition of the 24 cell means (per URL-level mean logit)
import itertools
fac = np.array([[FACT_SCHEME.index(sc), int(w), int(sl), int(p)] for sc, w, sl, p in FACT_CELLS])
an = []
for m in ['LightGBM', 'charCNN']:
    for s in SETS:
        for cls, nm in [(0, 'legitimate'), (1, 'phishing')]:
            mu = B1[(B1.model == m) & (B1.set == s) & (B1['class'] == nm)].set_index('cell').loc[CELLNAME].mean_logit.values
            # design: one-hot scheme (3) + binary factors, with main effects only vs. full model
            Xm = np.column_stack([np.ones(24), fac[:, 0] == 1, fac[:, 0] == 2, fac[:, 1], fac[:, 2], fac[:, 3]]).astype(float)
            beta, *_ = np.linalg.lstsq(Xm, mu, rcond=None); res = mu - Xm @ beta
            ss_tot = np.sum((mu - mu.mean()) ** 2); ss_main = ss_tot - np.sum(res ** 2)
            an.append({'model': m, 'set': s, 'class': nm, 'between_cell_SS': ss_tot, 'share_main_effects': ss_main / ss_tot if ss_tot > 0 else np.nan,
                       'share_interactions': 1 - ss_main / ss_tot if ss_tot > 0 else np.nan,
                       'effect_https_vs_http': -beta[1], 'effect_www': beta[3], 'effect_trailing_slash': beta[4], 'effect_path': beta[5]})
pd.DataFrame(an).to_csv(f'{RES}/rev_B2_factorial_anova.csv', index=False)

# ---------------------------------------------------------------- B3 residual FPs by path (URL-Phish legitimate)
s = 'URL-Phish'; y = P[s].label.values; haspath = np.array([len(url_parts(u)[3]) > 0 for u in P[s].url.values])
i_keep = CELLNAME.index('https|www|noslash|path'); i_host = CELLNAME.index('https|www|noslash|nopath')
b3 = []
for m in ['LightGBM', 'charCNN']:
    for sd in SEEDS:
        z0 = LOG[m, sd]['orig'][s]; fp = (z0 >= 0) & (y == 0)
        for grp, g in [('has path', haspath), ('no path', ~haspath), ('all', np.ones(len(y), bool))]:
            k = fp & g
            b3.append({'model': m, 'seed': sd, 'group': grp, 'n_FP': int(k.sum()),
                       'pct_corrected_template_path_kept': 100 * np.mean(LOG[m, sd][s][i_keep][k] < 0) if k.any() else np.nan,
                       'pct_corrected_template_host_only': 100 * np.mean(LOG[m, sd][s][i_host][k] < 0) if k.any() else np.nan})
pd.DataFrame(b3).to_csv(f'{RES}/rev_B3_residual_fp_by_path.csv', index=False)

# ---------------------------------------------------------------- B4 canonical-trained models: remaining FPs by path
b4 = []
for f in sorted(glob.glob(f'{ROOT}/predictions/charcnn/charCNN-canonical_s*.csv.gz')) + sorted(glob.glob(f'{ROOT}/predictions/transformer/DistilBERT-canonical_s*.csv.gz')) + \
        sorted(glob.glob(f'{ROOT}/predictions/charcnn_long/charCNN-*_s*.csv.gz')):
    d = pd.read_csv(f); d = d[(d.variant == 'orig') & (d.partition == s)].set_index('row_id').loc[P[s].row_id]
    z = d.logit.values
    for grp, g in [('has path', haspath), ('no path', ~haspath)]:
        k = (y == 0) & g
        b4.append({'file': os.path.basename(f), 'model': d.model.iloc[0], 'seed': int(d.seed.iloc[0]), 'group': grp, 'n_legit': int(k.sum()), 'FPR': np.mean(z[k] >= 0)})
pd.DataFrame(b4).to_csv(f'{RES}/rev_B4_canonical_residual_by_path.csv', index=False)
pd.DataFrame([{'cell': c, 'train_legit': support[c][0], 'train_phish': support[c][1]} for c in CELLNAME]).to_csv(f'{RES}/rev_B0_cell_training_support.csv', index=False)
print('done', round(time.time() - t0), 's')
