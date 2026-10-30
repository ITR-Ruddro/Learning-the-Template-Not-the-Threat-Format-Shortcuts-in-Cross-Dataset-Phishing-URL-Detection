"""Revision F: format audit of five labelled URL datasets with the urlformat-audit package (v0.1.0), and transfer of the
format-only rule (majority label per scheme|www|slash|path cell) between datasets.
Outputs: results/rev_F_audit_5datasets.csv, results/rev_F_format_rule_transfer.csv, results/rev_F_cell_profile.csv"""
import json, numpy as np, pandas as pd
from urlformat_audit import audit, compare, format_features
from common import DATA, RES

import os
X = os.environ.get('CP_EXTRA', os.path.join(DATA, 'extra'))   # see README: data/extra/pdd/..., data/extra/Using-machine-learning-to-detect-malicious-URLs/...
ds = {}
ph = pd.read_csv(f'{DATA}/phiusiil.csv', usecols=['URL', 'label']); ds['PhiUSIIL'] = (ph.URL.astype(str).tolist(), ph.label.astype(int).values)
up = pd.read_csv(f'{DATA}/urlphish.csv', usecols=['url', 'label']); ds['URL-Phish'] = (up.url.astype(str).tolist(), up.label.astype(int).values)
lp = pd.read_csv(f'{DATA}/legitphish.csv', usecols=['URL', 'ClassLabel']); lp = lp[lp.ClassLabel.notna()]
ds['LegitPhish'] = (lp.URL.astype(str).tolist(), (1 - lp.ClassLabel.astype(int)).values)
leg = json.load(open(f'{X}/pdd/input/data_legitimate_36400.json')); phi = json.load(open(f'{X}/pdd/input/data_phishing_37175.json'))
ds['Sahingoz2019'] = ([str(u) for u in leg + phi], np.array([0] * len(leg) + [1] * len(phi)))
fz = pd.read_csv(f'{X}/Using-machine-learning-to-detect-malicious-URLs/data/data.csv', dtype=str, keep_default_na=False)
ds['MaliciousURL-GH'] = (fz.url.tolist(), (fz.label.str.strip() == 'bad').astype(int).values)

rows, prof = [], []
for name, (u, y) in ds.items():
    r = audit(u, y, name=name)
    ct = r.class_table
    rows.append({'dataset': name, 'n': r.n, 'pct_positive': 100 * y.mean(),
                 **{f'neg_{c}': ct.loc['negative (0)', c] for c in ['scheme=https', 'scheme=http', 'scheme=none', 'www', 'trailing_slash', 'path', 'ip_host']},
                 **{f'pos_{c}': ct.loc['positive (1)', c] for c in ['scheme=https', 'scheme=http', 'scheme=none', 'www', 'trailing_slash', 'path', 'ip_host']},
                 'format_rule_MCC': r.format_rule['mcc'], 'format_rule_BalAcc': r.format_rule['balanced_accuracy'], 'format_rule_F1': r.format_rule['f1_positive'],
                 'template_rule_MCC': r.template_rule['mcc'], 'trivial_F1': r.trivial['f1_positive'], 'NMI_cell_label': r.nmi_cell_label,
                 'n_cells_used': len(r.cell_table), 'pct_rows_in_pure_cells_ge99': 100 * r.cell_table.loc[r.cell_table.purity >= 0.99, 'n'].sum() / r.n,
                 'n_flags': len(r.flags)})
    c = r.cell_table.reset_index(); c.insert(0, 'dataset', name); prof.append(c)
    print(name, round(r.format_rule['mcc'], 3), round(r.nmi_cell_label, 3), flush=True)
pd.DataFrame(rows).to_csv(f'{RES}/rev_F_audit_5datasets.csv', index=False)
pd.concat(prof).to_csv(f'{RES}/rev_F_cell_profile.csv', index=False)
tr = []
for a, (ua, ya) in ds.items():
    for b, (ub, yb) in ds.items():
        if a == b: continue
        r = compare(ua, ya, ub, yb)
        tr.append({'train': a, 'test': b, 'MCC': r['format_rule_transfer']['mcc'], 'F1': r['format_rule_transfer']['f1_positive'],
                   'FPR': r['format_rule_transfer']['fpr'], 'TPR': r['format_rule_transfer']['tpr'], 'trivial_F1': r['trivial_target']['f1_positive'],
                   'pct_test_in_unseen_cells': 100 * r['share_target_in_unseen_cells']})
pd.DataFrame(tr).to_csv(f'{RES}/rev_F_format_rule_transfer.csv', index=False)
pd.set_option('display.width', 250)
print(pd.DataFrame(rows)[['dataset', 'n', 'pct_positive', 'format_rule_MCC', 'format_rule_BalAcc', 'template_rule_MCC', 'NMI_cell_label', 'pct_rows_in_pure_cells_ge99']].round(3).to_string(index=False))
print(pd.DataFrame(tr).round(3).to_string(index=False))
