"""Step 0: scheme/format audit of raw URL strings, per dataset x class x cohort.
Usage: python step0_audit.py <data_dir> <splits_dir> <out_dir>"""
import sys, re, json, hashlib, numpy as np, pandas as pd
from urllib.parse import urlsplit
D, SP, OUT = sys.argv[1:4]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()

SCHEME = re.compile(r'^([A-Za-z][A-Za-z0-9+.-]*)://')
IPV4 = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')

def host_of(u):
    s = u if SCHEME.match(u) else 'http://' + u
    try:
        net = urlsplit(s).netloc
    except ValueError:
        net = s.split('/')[2] if s.count('/') >= 2 else ''
    net = net.rsplit('@', 1)[-1]
    return net

def feats(urls):
    u = pd.Series(urls).astype(str).str.strip()
    m = u.str.extract(SCHEME)[0].str.lower()
    host = u.map(host_of)
    hostname = host.str.replace(r'^\[.*\]', '[v6]', regex=True).str.split(':').str[0]
    port = host.str.contains(r'^[^\[]*:\d+$') | host.str.contains(r'\]:\d+$')
    rest = u.where(m.isna(), u.str.replace(SCHEME, '', regex=True))
    return pd.DataFrame({
        'https': (m == 'https'), 'http': (m == 'http'),
        'other_scheme': m.notna() & ~m.isin(['http', 'https']), 'no_scheme': m.isna(),
        'www': hostname.str.lower().str.startswith('www.'),
        'trailing_slash': u.str.endswith('/'),
        'uppercase_any': u.str.contains(r'[A-Z]'),
        'uppercase_host': hostname.str.contains(r'[A-Z]'),
        'explicit_port': port,
        'ip_host': hostname.str.match(IPV4) | hostname.eq('[v6]'),
        'path_beyond_root': rest.str.contains(r'/.') ,
        'query': u.str.contains(r'\?'),
        'length': u.str.len(),
    })

rows, meta = [], {}
def audit(name, cohort, urls, labels):
    f = feats(urls)
    for cls, nm in [(0, 'legitimate'), (1, 'phishing'), (None, 'all')]:
        sel = np.ones(len(f), bool) if cls is None else (np.asarray(labels) == cls)
        g = f[sel]
        r = {'dataset': name, 'cohort': cohort, 'class': nm, 'n': int(sel.sum())}
        for c in g.columns:
            if c != 'length':
                r['pct_' + c] = round(100 * g[c].mean(), 2)
        r['median_length'] = float(g['length'].median())
        r['p10_length'] = float(g['length'].quantile(.1)); r['p90_length'] = float(g['length'].quantile(.9))
        rows.append(r)

# ---- PhiUSIIL (label 1 = phishing)
p = f'{D}/phiusiil.csv'; meta['phiusiil_sha256'] = sha(p)
ph = pd.read_csv(p, usecols=['URL', 'label'])
parts = np.load(f'{SP}/source_partitions.npz')
# data/phiusiil.csv is written by prepare_data.py in the project coding (1 = phishing); check it from the data:
# every PhiUSIIL legitimate URL has the form https://www.<host>.
_t = ph.URL.astype(str).str.strip().str.match(r'^https://www\.[a-z0-9.-]+$')
assert _t[ph.label == 0].mean() > 0.99, 'phiusiil.csv is not in project coding; run prepare_data.py'
lab = ph.label.values
audit('PhiUSIIL', 'full release', ph.URL, lab)
for k, nm in [('source_train', 'train'), ('source_validation', 'validation'), ('source_test', 'test')]:
    i = parts[k]; audit('PhiUSIIL', nm, ph.URL.values[i], lab[i])
meta['phiusiil_partition_sizes'] = {k: int(parts[k].size) for k in parts}

# ---- URL-Phish (label 1 = phishing in release)
p = f'{D}/urlphish.csv'; meta['urlphish_sha256'] = sha(p)
up = pd.read_csv(p, usecols=['url', 'label'])
sel = np.load(f'{SP}/urlphish_selection.npz')['corrected_full_row_indices']
audit('URL-Phish', 'full release', up.url, up.label.values)
audit('URL-Phish', 'evaluated cohort', up.url.values[sel], up.label.values[sel])
meta['urlphish_cohort_size'] = int(sel.size)
# ---- LegitPhish (ClassLabel 1 = legitimate in release -> project label = 1 - ClassLabel); drop the unlabeled row
p = f'{D}/legitphish.csv'; meta['legitphish_sha256'] = sha(p)
lp = pd.read_csv(p, usecols=['URL', 'ClassLabel'])
meta['legitphish_unlabeled_rows_dropped'] = int(lp.ClassLabel.isna().sum())
lp = lp[lp.ClassLabel.notna()].reset_index(drop=True)
llab = (1 - lp.ClassLabel.values).astype(int)
lsel = np.load(f'{SP}/legitphish_selection.npz')['corrected_full_row_indices']
audit('LegitPhish', 'full release', lp.URL, llab)
audit('LegitPhish', 'evaluated cohort', lp.URL.values[lsel], llab[lsel])
meta['legitphish_cohort_size'] = int(lsel.size)

df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/step0_scheme_audit.csv', index=False)
json.dump(meta, open(f'{OUT}/step0_scheme_audit_meta.json', 'w'), indent=1)
# example strings per dataset x class for the reader
ex = []
for name, u, l in [('PhiUSIIL', ph.URL, lab), ('URL-Phish', up.url.values[sel], up.label.values[sel]), ('LegitPhish', lp.URL.values[lsel], llab[lsel])]:
    s = pd.DataFrame({'url': np.asarray(u), 'label': np.asarray(l)})
    for c in (0, 1):
        for x in s[s.label == c].sample(8, random_state=42).url: ex.append({'dataset': name, 'label': c, 'url': x})
pd.DataFrame(ex).to_csv(f'{OUT}/step0_examples.csv', index=False)
print(json.dumps(meta, indent=1))
