"""Step 1: LightGBM reference (fixed untuned hyper-parameters, 15 URL features, seeds 42-44) + IsHTTPS 0<->1 flip.
Writes predictions/lgbm_logits.csv.gz and results/step1_*.csv"""
import time, json, numpy as np, pandas as pd, lightgbm, sklearn
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
from common import *
from evaluate import perf_table, cf_table, attach_groups

t0 = time.time()
P = load_all()
F = {k: pd.read_pickle(f"{RES}/cache/feat_{k.replace(' ', '_')}_orig.pkl") for k in P}
for k in P: assert (F[k].row_id.values == P[k].row_id.values).all()
enc = ColumnTransformer([('numeric', StandardScaler(), NUMERIC),
                         ('tld', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64), ['TLD'])],
                        sparse_threshold=1.0)
enc.fit(F['train'][FEATURES])
X = {k: sparse.csr_matrix(enc.transform(F[k][FEATURES])) for k in P}


def flipX(k):
    f = F[k][FEATURES].copy(); f['IsHTTPS'] = 1 - f['IsHTTPS']
    return sparse.csr_matrix(enc.transform(f))


Xf = {k: flipX(k) for k in ['val'] + SETS}
rows = []
fit_times = {}
for sd in SEEDS:
    m = LGBMClassifier(n_estimators=100, learning_rate=.05, max_depth=5, num_leaves=31, subsample=.8, subsample_freq=0,
                       colsample_bytree=.8, class_weight='balanced', random_state=sd, n_jobs=2, verbosity=-1,
                       deterministic=True, force_col_wise=True)
    ts = time.time(); m.fit(X['train'], P['train'].label.values); fit_times[sd] = round(time.time() - ts, 1)
    for k in ['val'] + SETS:
        for var, XX in [('orig', X[k]), ('IsHTTPS_flip', Xf[k])]:
            z = m.booster_.predict(XX, raw_score=True)
            rows.append(pd.DataFrame({'row_id': P[k].row_id, 'dataset': P[k].dataset, 'partition': k, 'label': P[k].label,
                                      'logit': z, 'variant': var, 'seed': sd, 'model': 'LightGBM',
                                      'changed': True if var != 'orig' else False}))
L = pd.concat(rows, ignore_index=True)
os.makedirs(f'{ROOT}/predictions', exist_ok=True)
L.drop(columns=['changed']).to_csv(f'{ROOT}/predictions/lgbm_logits.csv.gz', index=False, float_format='%.6f')
L = attach_groups(L, P)

perf = perf_table(L, 'LightGBM', 'orig'); perf.to_csv(f'{RES}/step1_lgbm_performance.csv', index=False)
cf = []
for s in SETS:  # flip direction split: https->http (IsHTTPS 1->0) and http->https (0->1)
    pass
cfall = cf_table(L, 'LightGBM', 'IsHTTPS_flip')
# by direction: mark 'changed' only for one direction at a time
isH = pd.concat([F[k][['row_id']].assign(partition=k, IsHTTPS=F[k].IsHTTPS.values) for k in ['val'] + SETS])
L2 = L.merge(isH, on=['row_id', 'partition'], how='left')
for d, nm in [(1, 'https->http'), (0, 'http->https')]:
    L3 = L2.copy(); L3['changed'] = (L3.IsHTTPS == d)
    t = cf_table(L3, 'LightGBM', 'IsHTTPS_flip'); t['direction'] = nm; cf.append(t)
cfall['direction'] = 'both'
cft = pd.concat([cfall] + cf, ignore_index=True)
cft.to_csv(f'{RES}/step1_lgbm_IsHTTPS_flip.csv', index=False)
json.dump({'lightgbm': lightgbm.__version__, 'sklearn': sklearn.__version__, 'fit_seconds': fit_times,
           'total_seconds': round(time.time() - t0), 'n_encoded_features': int(X['train'].shape[1]),
           'feature_source': 're-extracted from raw URLs'},
          open(f'{RES}/step1_run_info.json', 'w'), indent=1)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(perf[['set', 'n', 'F1', 'F1_lo', 'F1_hi', 'PR_AUC', 'FPR', 'FNR', 'TPR_at_valFPR1', 'achFPR_valthr', 'TPR_at_oracleFPR1', 'Brier', 'ECE15']].round(4).to_string(index=False))
print(cft[['direction', 'set', 'cls', 'n_changed', 'orig_error_rate', 'mean_dlogit', 'mean_dlogit_lo', 'mean_dlogit_hi', 'flip_rate', 'pct_errors_corrected', 'pct_correct_broken']].round(4).to_string(index=False))
print('seconds', round(time.time() - t0))
