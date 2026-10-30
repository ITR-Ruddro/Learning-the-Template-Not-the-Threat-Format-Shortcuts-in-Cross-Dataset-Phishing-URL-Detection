"""Step 0 diagnostic: how well does a pure string-format rule separate the classes?
Rule: predict LEGITIMATE iff URL matches the PhiUSIIL-legitimate template ^https://www\.<host>$ (no path, slash, query, port, uppercase).
Also reports single-cue rules. Usage: python step0_template_rule.py <data_dir> <splits_dir> <out_dir>"""
import sys, re, numpy as np, pandas as pd
D, SP, OUT = sys.argv[1:4]
TPL = re.compile(r'^https://www\.[a-z0-9.-]+$')
ph = pd.read_csv(f'{D}/phiusiil.csv', usecols=['URL', 'label'])
parts = np.load(f'{SP}/source_partitions.npz')
up = pd.read_csv(f'{D}/urlphish.csv', usecols=['url', 'label'])
sel = np.load(f'{SP}/urlphish_selection.npz')['corrected_full_row_indices']
lp = pd.read_csv(f'{D}/legitphish.csv', usecols=['URL', 'ClassLabel']); lp = lp[lp.ClassLabel.notna()].reset_index(drop=True)
lsel = np.load(f'{SP}/legitphish_selection.npz')['corrected_full_row_indices']
sets = {'PhiUSIIL test': (ph.URL.values[parts['source_test']], ph.label.values[parts['source_test']]),
        'URL-Phish cohort': (up.url.values[sel], up.label.values[sel]),
        'LegitPhish cohort': (lp.URL.values[lsel], (1 - lp.ClassLabel.values[lsel]).astype(int))}
rules = {
    'template ^https://www.<host>$ -> legit': lambda u: u.str.match(TPL),
    'scheme https -> legit': lambda u: u.str.lower().str.startswith('https://'),
    'no trailing slash -> legit': lambda u: ~u.str.endswith('/'),
    'host starts www. -> legit': lambda u: u.str.lower().str.contains(r'^[a-z]+://www\.'),
}
out = []
for sname, (u, y) in sets.items():
    u = pd.Series(u).astype(str).str.strip(); y = np.asarray(y)
    for rname, f in rules.items():
        legit = f(u).values; pred = (~legit).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        out.append({'set': sname, 'rule': rname, 'n': len(y),
                    'pct_legit_matching': round(100 * legit[y == 0].mean(), 2),
                    'pct_phish_matching': round(100 * legit[y == 1].mean(), 2),
                    'accuracy': round((pred == y).mean(), 4),
                    'F1_phishing': round(2 * tp / (2 * tp + fp + fn), 4),
                    'FPR_legit': round(fp / (y == 0).sum(), 4), 'TPR': round(tp / (y == 1).sum(), 4)})
df = pd.DataFrame(out); df.to_csv(f'{OUT}/step0_format_rule_baselines.csv', index=False)
pd.set_option('display.width', 250); print(df.to_string(index=False))
