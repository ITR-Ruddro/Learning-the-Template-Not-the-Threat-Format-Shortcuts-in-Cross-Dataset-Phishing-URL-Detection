"""Extract the 15 lexical features from the raw URL of every partition (cached in results/cache/ for step1_lgbm.py)."""
import time, pandas as pd
from common import *
t0 = time.time(); P = load_all(); os.makedirs(f'{RES}/cache', exist_ok=True)
for k, d in P.items():
    f = feature_frame(d.url.values); f.insert(0, 'row_id', d.row_id.values)
    f.to_pickle(f"{RES}/cache/feat_{k.replace(' ', '_')}_orig.pkl")
print('features cached in results/cache/, seconds', round(time.time() - t0))
