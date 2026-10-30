"""Shared data loading, URL variants, feature extraction, metrics and grouped bootstrap.
Paths: data/ (prepared by src/prepare_data.py) and splits/ at the repository root; override with CP_DATA / CP_SPLITS."""
import os, re, json, hashlib, numpy as np, pandas as pd
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('CP_DATA', os.path.join(ROOT, 'data'))
SPL = os.environ.get('CP_SPLITS', os.path.join(ROOT, 'splits'))
RES = os.path.join(ROOT, 'results')
SEEDS = [42, 43, 44]
SETS = ['PhiUSIIL test', 'URL-Phish', 'LegitPhish']


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load_all():
    """Returns dict partition -> DataFrame(row_id, dataset, url, label). label 1 = phishing.
    Partitions: train, val, PhiUSIIL test, URL-Phish, LegitPhish (evaluated cohorts)."""
    ph = pd.read_csv(f'{DATA}/phiusiil.csv', usecols=['URL', 'label'])
    parts = np.load(f'{SPL}/source_partitions.npz')
    up = pd.read_csv(f'{DATA}/urlphish.csv', usecols=['url', 'label'])
    usel = np.load(f'{SPL}/urlphish_selection.npz')['corrected_full_row_indices']
    lp = pd.read_csv(f'{DATA}/legitphish.csv', usecols=['URL', 'ClassLabel'])
    lp = lp[lp.ClassLabel.notna()].reset_index(drop=True)
    lsel = np.load(f'{SPL}/legitphish_selection.npz')['corrected_full_row_indices']
    out = {}
    for k, nm in [('source_train', 'train'), ('source_validation', 'val'), ('source_test', 'PhiUSIIL test')]:
        i = parts[k]
        out[nm] = pd.DataFrame({'row_id': [f'PhiUSIIL:{j}' for j in i], 'dataset': 'PhiUSIIL',
                                'url': ph.URL.values[i].astype(str), 'label': ph.label.values[i].astype(int)})
    out['URL-Phish'] = pd.DataFrame({'row_id': [f'URL-Phish:{j}' for j in usel], 'dataset': 'URL-Phish',
                                     'url': up.url.values[usel].astype(str), 'label': up.label.values[usel].astype(int)})
    out['LegitPhish'] = pd.DataFrame({'row_id': [f'LegitPhish:{j}' for j in lsel], 'dataset': 'LegitPhish',
                                      'url': lp.URL.values[lsel].astype(str),
                                      'label': (1 - lp.ClassLabel.values[lsel]).astype(int)})
    for k, d in out.items():
        d['group'] = pd.factorize(d.url.str.strip())[0]
    assert [len(out[k]) for k in ['train', 'val', 'PhiUSIIL test', 'URL-Phish', 'LegitPhish']] == [150931, 37716, 47148, 116126, 71767]
    return out


# ------------------------------------------------------------------ URL variants (E2)
SCH = re.compile(r'^([A-Za-z][A-Za-z0-9+.-]*)://')


def has_scheme(u):
    return SCH.match(u.strip()) is not None


def v_swap(u):
    u = u.strip(); m = SCH.match(u)
    if not m: return u
    s = m.group(1).lower()
    if s == 'https': return 'http://' + u[m.end():]
    if s == 'http': return 'https://' + u[m.end():]
    return u


def v_noscheme(u):
    return SCH.sub('', u.strip(), count=1)


def _split_host(rest):
    i = rest.find('/'); j = rest.find('?'); k = rest.find('#')
    cut = min([x for x in (i, j, k) if x >= 0], default=len(rest))
    return rest[:cut], rest[cut:]


def v_canonical(u):
    """strip scheme, leading 'www.', trailing '/', lowercase host"""
    rest = v_noscheme(u)
    host, tail = _split_host(rest)
    host = host.lower()
    if host.startswith('www.'): host = host[4:]
    out = host + tail
    return out.rstrip('/') if out.rstrip('/') else out


def v_noslash(u):
    u = u.strip()
    return u.rstrip('/') if SCH.sub('', u).rstrip('/') else u


def v_template(u):
    """project onto the PhiUSIIL-legitimate template https://www.<host>[path] : scheme->https, add www., drop trailing '/'"""
    rest = v_noscheme(u)
    host, tail = _split_host(rest)
    host = host.lower()
    if not host.startswith('www.') and not re.match(r'^\d{1,3}(\.\d{1,3}){3}(:\d+)?$', host):
        host = 'www.' + host
    out = 'https://' + host + tail
    return out.rstrip('/') if (host + tail).rstrip('/') else out


def v_hostonly(u):
    """format-neutral AND path-free: lower-cased host without scheme, 'www.' or anything after the host"""
    host, _ = _split_host(v_noscheme(u))
    host = host.lower()
    return host[4:] if host.startswith('www.') else host


def url_parts(u):
    """(scheme or None, host without leading www. (case kept), had_www, tail without trailing '/', had_trailing_slash)"""
    u = u.strip(); m = SCH.match(u)
    sch = m.group(1).lower() if m else None
    rest = u[m.end():] if m else u
    host, tail = _split_host(rest)
    www = host.lower().startswith('www.')
    if www: host = host[4:]
    slash = (host + tail).endswith('/') and len((host + tail).rstrip('/')) > 0
    tail = tail.rstrip('/') if slash else tail
    return sch, host, www, tail, slash


FACT_SCHEME = ['https', 'http', 'none']


def factorial_variant(u, scheme, www, slash, path):
    """rebuild a URL with chosen scheme (https/http/none), www prefix (bool), trailing slash (bool), path kept (bool)"""
    _, host, _, tail, _ = url_parts(u)
    s = '' if scheme == 'none' else scheme + '://'
    body = ('www.' if www else '') + host + (tail if path else '')
    return s + body + ('/' if slash else '')


FACT_CELLS = [(sc, w, sl, p) for sc in FACT_SCHEME for w in (True, False) for sl in (True, False) for p in (True, False)]


VARIANTS = {'swap': v_swap, 'noscheme': v_noscheme, 'canonical': v_canonical, 'noslash': v_noslash, 'template': v_template}

# ------------------------------------------------------------------ 15 lexical features
try:
    import tldextract
    EXT = tldextract.TLDExtract(include_psl_private_domains=True, suffix_list_urls=())
except Exception:  # text-model notebooks do not need this
    EXT = None
IP = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
FEATURES = ['URLLength', 'DomainLength', 'IsDomainIP', 'TLD', 'TLDLength', 'NoOfSubDomain', 'NoOfLettersInURL',
            'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL', 'NoOfEqualsInURL', 'NoOfQMarkInURL',
            'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL', 'IsHTTPS']
NUMERIC = [c for c in FEATURES if c != 'TLD']


def feats(url):
    url = str(url).strip(); L = max(len(url), 1)
    p = urlparse(url if "//" in url else f"http://{url}"); e = EXT(url)
    try:
        hn = p.hostname or ""
    except ValueError:
        hn = ""
    let = sum(("a" <= c <= "z") or ("A" <= c <= "Z") for c in url); dig = sum("0" <= c <= "9" for c in url)
    eq, q, am = url.count('='), url.count('?'), url.count('&'); sp = len(url) - let - dig
    return (len(url), len(e.fqdn), int(bool(IP.match(hn))), e.suffix if e.suffix else 'missing', len(e.suffix),
            len(e.subdomain.split('.')) if e.subdomain else 0, let, let / L, dig, dig / L, eq, q, am, sp - (eq + q + am),
            int(p.scheme.lower() == 'https'))


def feature_frame(urls):
    return pd.DataFrame([feats(u) for u in urls], columns=FEATURES)


# ------------------------------------------------------------------ metrics
def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def _w(n, w):
    return np.ones(n) if w is None else w


def wap(score, y, w=None, order=None):
    """weighted average precision (sklearn definition, ties grouped)"""
    w = _w(len(y), w)
    if order is None: order = np.argsort(-score, kind='mergesort')
    s, yy, ww = score[order], y[order], w[order]
    tp = np.cumsum(ww * yy); fp = np.cumsum(ww * (1 - yy))
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp, fp = tp[last], fp[last]
    P = tp[-1]
    if P <= 0: return np.nan
    prec = tp / np.maximum(tp + fp, 1e-300); rec = tp / P
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def thr_at_fpr(score_neg, w=None, fpr=0.01):
    """smallest threshold t such that weighted fraction of negatives with score >= t is <= fpr (predict positive if score >= t)"""
    w = _w(len(score_neg), w)
    o = np.argsort(-score_neg, kind='mergesort'); s, ww = score_neg[o], w[o]
    cum = np.cumsum(ww) / ww.sum()
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    ok = last[cum[last] <= fpr]
    if len(ok) == 0: return np.nextafter(s[0], np.inf)  # nothing allowed above
    k = ok[-1]  # can admit everything with score >= s[k]
    return s[k]


def ece15(p, y, w=None):
    w = _w(len(y), w)
    b = np.minimum((p * 15).astype(int), 14)
    W = np.bincount(b, w, 15); P = np.bincount(b, w * p, 15); Y = np.bincount(b, w * y, 15)
    m = W > 0
    return float(np.sum(np.abs(P[m] - Y[m])) / W.sum())


def metrics(z, y, w=None, t_val=None, order=None):
    """z: raw logit (margin). t_val: threshold chosen on source validation for FPR<=1%."""
    w = _w(len(y), w); p = sig(z); pred = z >= 0
    tp = np.sum(w * (pred & (y == 1))); fp = np.sum(w * (pred & (y == 0))); fn = np.sum(w * (~pred & (y == 1)))
    neg = np.sum(w * (y == 0)); pos = np.sum(w * (y == 1))
    r = {'F1': 2 * tp / max(2 * tp + fp + fn, 1e-300), 'PR_AUC': wap(z, y, w, order), 'FPR': fp / neg, 'FNR': fn / pos,
         'Brier': float(np.sum(w * (p - y) ** 2) / w.sum()), 'ECE15': ece15(p, y, w)}
    if t_val is not None:
        sel = z >= t_val
        r['TPR_at_valFPR1'] = np.sum(w * (sel & (y == 1))) / pos
        r['achFPR_valthr'] = np.sum(w * (sel & (y == 0))) / neg
    m = y == 0
    t_or = thr_at_fpr(z[m], w[m], 0.01)
    sel = z >= t_or
    r['TPR_at_oracleFPR1'] = np.sum(w * (sel & (y == 1))) / pos
    r['achFPR_oraclethr'] = np.sum(w * (sel & (y == 0))) / neg
    return r


def boot_weights(group, y, B=1000, seed=0):
    """class-stratified cluster bootstrap with exact-URL groups as the unit; yields row-weight vectors"""
    rng = np.random.default_rng(seed)
    idx = {}
    for c in (0, 1):
        g = np.unique(group[y == c])
        idx[c] = g
    ng = group.max() + 1
    for _ in range(B):
        cnt = np.zeros(ng)
        for c in (0, 1):
            g = idx[c]
            np.add.at(cnt, rng.choice(g, len(g), replace=True), 1)
        yield cnt[group]


def summarize(point_by_seed, boot_by_seed):
    """point_by_seed: list of dicts; boot_by_seed: list (seed) of list (B) of dicts -> mean over seeds + percentile CI of seed-mean"""
    keys = point_by_seed[0].keys(); out = {}
    for k in keys:
        out[k] = float(np.mean([d[k] for d in point_by_seed]))
        bs = np.mean([[b[k] for b in bl] for bl in boot_by_seed], axis=0)
        out[k + '_lo'], out[k + '_hi'] = [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]
        out[k + '_seed_sd'] = float(np.std([d[k] for d in point_by_seed], ddof=1)) if len(point_by_seed) > 1 else np.nan
    return out
