"""Step 4 support: LightGBM logits under the five string-level URL variants (features re-extracted from the edited
string). Refits the same deterministic models as step1_lgbm.py (verified identical logits) and appends to
predictions/lgbm_variant_logits.csv.gz"""
import time, numpy as np, pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
from common import *

t0 = time.time(); P = load_all()
F = {k: pd.read_pickle(f"{RES}/cache/feat_{k.replace(' ', '_')}_orig.pkl") for k in P}
enc = ColumnTransformer([('numeric', StandardScaler(), NUMERIC),
                         ('tld', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64), ['TLD'])], sparse_threshold=1.0)
enc.fit(F['train'][FEATURES])
VX, CH = {}, {}
for k in SETS:
    base = np.array([u.strip() for u in P[k].url], dtype=object)
    for vn, f in VARIANTS.items():
        vu = np.array([f(u) for u in P[k].url], dtype=object)
        VX[k, vn] = sparse.csr_matrix(enc.transform(feature_frame(vu)[FEATURES])); CH[k, vn] = vu != base
ref = pd.read_csv(f'{ROOT}/predictions/lgbm_logits.csv.gz'); ref = ref[ref.variant == 'orig']
rows = []
for sd in SEEDS:
    m = LGBMClassifier(n_estimators=100, learning_rate=.05, max_depth=5, num_leaves=31, subsample=.8, subsample_freq=0,
                       colsample_bytree=.8, class_weight='balanced', random_state=sd, n_jobs=2, verbosity=-1,
                       deterministic=True, force_col_wise=True)
    m.fit(sparse.csr_matrix(enc.transform(F['train'][FEATURES])), P['train'].label.values)
    for k in SETS:
        z0 = m.booster_.predict(sparse.csr_matrix(enc.transform(F[k][FEATURES])), raw_score=True)
        r = ref[(ref.seed == sd) & (ref.partition == k)]
        assert np.allclose(r.logit.values, z0, atol=1e-5), 'refit differs from step 1'
        for vn in VARIANTS:
            rows.append(pd.DataFrame({'row_id': P[k].row_id, 'dataset': P[k].dataset, 'partition': k, 'label': P[k].label,
                                      'logit': m.booster_.predict(VX[k, vn], raw_score=True), 'variant': vn, 'seed': sd,
                                      'model': 'LightGBM', 'changed': CH[k, vn]}))
pd.concat(rows).to_csv(f'{ROOT}/predictions/lgbm_variant_logits.csv.gz', index=False, float_format='%.6f')
print('done', round(time.time() - t0), 's')
