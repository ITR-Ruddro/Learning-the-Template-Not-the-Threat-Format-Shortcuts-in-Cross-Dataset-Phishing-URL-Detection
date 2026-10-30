"""Prepare the three public datasets for the experiments (run once).
Reads the original downloads and writes the minimal files used by every script:
  data/phiusiil.csv    columns URL,label  (label 1 = phishing; the PhiUSIIL release codes 1 = legitimate, so it is inverted)
  data/urlphish.csv    columns url,label  (label 1 = phishing, as released)
  data/legitphish.csv  columns URL,ClassLabel (as released: ClassLabel 1 = legitimate, one row unlabeled; the code converts and drops it)
Two PhiUSIIL training URLs contain double-encoded UTF-8 ("mojibake") in the release; they are repaired with a cp1252 round trip,
exactly as in the paper's copy. A content hash over (URL, label) is checked against the paper's data.

Usage: python prepare_data.py --phiusiil PhiUSIIL_Phishing_URL_Dataset.csv --urlphish URL-Phish.csv --legitphish "LegitPhish Dataset.csv"
"""
import argparse, hashlib, os, pandas as pd

EXPECTED = {'phiusiil': ('1b6274802878ed9e09f4477fe4f941b3179377ed02ac13ed41c56d5036556f01', 235795),
            'urlphish': ('e0d305f8f994b39d6a26e8f5d6fc0d903431e486a9cd70358156fabd78b8eb10', 116600),
            'legitphish': ('81b5229a03b7f51739186c9ca614722238b2a74e32379b92401bff5bd0d65d3d', 101219)}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def chash(a, b):
    h = hashlib.sha256()
    for u, l in zip(a, b): h.update(f'{u}\t{l}\n'.encode())
    return h.hexdigest()


def fix_mojibake(u):
    try:
        return u.encode('cp1252').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return u


def check(name, a, b):
    h, n = EXPECTED[name]; got = chash(a, b)
    ok = got == h and len(a) == n
    print(f'{name:10s} rows={len(a):,}  content hash {"OK" if ok else "DIFFERS: " + got[:16]}')
    if not ok:
        print(f'  WARNING: {name} differs from the copy used in the paper (expected {n:,} rows, hash {h[:16]}...). '
              'Results may differ; the saved split indices assume the same row order.')


ap = argparse.ArgumentParser()
ap.add_argument('--phiusiil', required=True); ap.add_argument('--urlphish', required=True); ap.add_argument('--legitphish', required=True)
ap.add_argument('--out', default=os.path.join(ROOT, 'data'))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

ph = pd.read_csv(a.phiusiil, usecols=['URL', 'label'], dtype=str, keep_default_na=False, encoding='utf-8-sig')
tpl = ph.URL.str.strip().str.match(r'^https://www\.[a-z0-9.-]+$')
if tpl[ph.label == '1'].mean() > tpl[ph.label == '0'].mean():  # release coding: 1 = legitimate
    ph['label'] = ph.label.map({'1': '0', '0': '1'}); print('PhiUSIIL: label coding converted to 1 = phishing')
ph['URL'] = ph.URL.map(fix_mojibake)
check('phiusiil', ph.URL, ph.label); ph.to_csv(os.path.join(a.out, 'phiusiil.csv'), index=False)

up = pd.read_csv(a.urlphish, usecols=['url', 'label'], dtype=str, keep_default_na=False)
check('urlphish', up.url, up.label); up.to_csv(os.path.join(a.out, 'urlphish.csv'), index=False)

lp = pd.read_csv(a.legitphish, usecols=['URL', 'ClassLabel'], dtype=str, keep_default_na=False)
lp['ClassLabel'] = lp.ClassLabel.map(lambda x: x if x == '' else str(int(float(x))))
check('legitphish', lp.URL, lp.ClassLabel); lp.to_csv(os.path.join(a.out, 'legitphish.csv'), index=False)
print('written to', a.out)
