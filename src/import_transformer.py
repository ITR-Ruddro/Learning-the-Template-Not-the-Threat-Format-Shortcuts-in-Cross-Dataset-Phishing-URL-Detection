"""Import GPU-notebook transformer predictions into predictions/transformer/.
Checks row order and labels against the saved cohorts. If a run was trained with the original PhiUSIIL label coding
(1 = legitimate; this happened in the paper's own run), its PhiUSIIL labels are exactly inverted and its logit is the
legitimate-class logit; because BCE(z, 1-y) = BCE(-z, y), the logit is negated to give the phishing logit of the same
model. Runs made with the current notebook (which converts the coding itself) are imported unchanged.
Usage: python import_transformer.py <folder with transformer_*.csv.gz>"""
import sys, glob, json, numpy as np, pandas as pd
from common import *
SRC = sys.argv[1]; OUTSUB = sys.argv[2] if len(sys.argv) > 2 else 'transformer'; ONLY = sys.argv[3].split(',') if len(sys.argv) > 3 else None
P = load_all(); rep = []; os.makedirs(f'{ROOT}/predictions/{OUTSUB}', exist_ok=True)
for f in sorted(glob.glob(f'{SRC}/transformer_*.csv.gz')):
    if '_attr_' in f or '_factorial_' in f: continue
    if ONLY and not any(f.endswith(o) for o in ONLY): continue
    d = pd.read_csv(f); out = []
    v0 = d[(d.partition == 'val') & (d.variant == 'orig')]
    inverted = bool((v0.label.values == 1 - P['val'].label.values).all())
    sign = -1.0 if inverted else 1.0
    for k in ['val'] + SETS:
        for v in (['orig'] if k == 'val' else ['orig'] + list(VARIANTS)):
            x = d[(d.partition == k) & (d.variant == v)].copy()
            assert (x.row_id.values == P[k].row_id.values).all()
            if P[k].dataset.iloc[0] == 'PhiUSIIL' and inverted:
                assert (x.label.values == 1 - P[k].label.values).all(), 'PhiUSIIL labels not exactly inverted'
            else:
                assert (x.label.values == P[k].label.values).all(), 'labels do not match the cohort'
            x['label'] = P[k].label.values; x['logit'] = sign * x.logit.values
            out.append(x)
    o = pd.concat(out); m0 = o.model.iloc[0]
    o['model'] = m0.replace('distilbert-base-cased', 'DistilBERT')
    name = f"{o.model.iloc[0]}_s{o.seed.iloc[0]}"
    o.to_csv(f'{ROOT}/predictions/{OUTSUB}/{name}.csv.gz', index=False, float_format='%.6f')
    v = o[(o.partition == 'val') & (o.variant == 'orig')]
    rep.append({'file': f.split('/')[-1], 'saved_as': f'predictions/{OUTSUB}/{name}.csv.gz', 'rows': len(o),
                'source_labels_inverted': inverted, 'val_acc_after_sign_fix': float(((v.logit >= 0) == v.label).mean()), 'sha256_upload': sha(f)})
    j = f.replace('.csv.gz', '_train_log.json'); json.dump(json.load(open(j)), open(f'{ROOT}/predictions/{OUTSUB}/{name}_train_log.json', 'w'), indent=1)
pd.DataFrame(rep).to_csv(f'{RES}/transformer_import_check_{OUTSUB}_{len(rep)}.csv', index=False); print(pd.DataFrame(rep).to_string())
