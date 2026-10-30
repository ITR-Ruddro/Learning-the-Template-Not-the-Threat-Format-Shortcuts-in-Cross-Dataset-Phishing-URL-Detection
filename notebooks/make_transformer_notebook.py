"""Builds transformer_gpu.ipynb (Colab/Kaggle) from the cell sources below."""
import json, sys, re

CELLS = []
md = lambda s: CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Transformer on raw URL text — GPU notebook (Steps 2b, 4, 5, 6)
Runs on Colab (T4/L4/A100) or Kaggle (T4/P100). Uses **exactly** the saved split and cohort indices of the char-CNN / LightGBM runs.

**You need these 6 files in one folder** (set `DATA_DIR` below; e.g. a Google Drive folder):
`data/phiusiil.csv`, `data/urlphish.csv`, `data/legitphish.csv` (written by `src/prepare_data.py`) and
`splits/source_partitions.npz`, `splits/urlphish_selection.npz`, `splits/legitphish_selection.npz` from this repository.

**Outputs** (to `OUT_DIR`): `transformer_<model>_s<seed>.csv.gz` with columns
`row_id, dataset, partition, label, logit, variant, seed, model, changed`, plus `*_train_log.json`,
and `transformer_attr_<model>_s42.csv.gz` (Integrated Gradients + occlusion scheme-span shares).
Upload the whole `OUT_DIR` folder back to the chat.

Run order: cell 1 → cell 7. Each (mode, seed) is skipped if its output file already exists, so the notebook can be resumed after a Colab disconnect.
""")
code(r"""
# 1. setup
!pip -q install "transformers>=4.48" captum tldextract 2>/dev/null
import os, re, json, time, random, hashlib, numpy as np, pandas as pd, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
assert torch.cuda.is_available(), 'Select a GPU runtime (Runtime > Change runtime type > GPU).'
""")
code(r"""
# 2. configuration — edit paths here
IN_COLAB = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False
if IN_COLAB:
    from google.colab import drive; drive.mount('/content/drive')
DATA_DIR = '/content/drive/MyDrive/confpaper_inputs'      # folder with the 6 input files
OUT_DIR  = '/content/drive/MyDrive/transformer_out'
MODEL_NAME = 'distilbert-base-cased'   # alternative: 'answerdotai/ModernBERT-base' (slower; needs transformers>=4.48)
MAX_LEN, EPOCHS, LR, BATCH, WARMUP = 128, 3, 2e-5, 64, 0.06
# Revision runs, in priority order (each ~45-55 min on a T4; the notebook skips finished runs, so it can be resumed):
RUNS = [('orig', 42),                                   # P1: correct label coding + factorial edits + attribution
        ('hostonly', 42),                               # P2: path-free format-neutral training
        ('canonical', 43), ('canonical', 44),           # P3: E4 seeds 43/44
        ('noscheme', 43), ('noscheme', 44), ('swapaug', 43), ('swapaug', 44), ('hostonly', 43), ('hostonly', 44)]
FACTORIAL_N = 10000      # URLs per (set, class) for the 24-cell factorial edit (orig models only)
DO_ATTRIBUTION = True    # Step 5 / E3 on the ('orig', 42) model
ATTR_N = 5000            # max URLs per (set, FP/TN) group
MODEL_LOGIT_SIGN = 1     # set to -1 ONLY for a model trained with 1 = legitimate (the 28 Oct 2026 run)
os.makedirs(OUT_DIR, exist_ok=True)
SHORT = MODEL_NAME.split('/')[-1]
""")
code(r"""
# 3. data: identical partitions / cohorts (assert-checked)
def chash(a, b):
    h = hashlib.sha256()
    for u, l in zip(a, b): h.update(f'{u}\t{l}\n'.encode())
    return h.hexdigest()
EXPECT = {'phiusiil.csv': (['URL','label'], '1b6274802878ed9e09f4477fe4f941b3179377ed02ac13ed41c56d5036556f01'),
          'urlphish.csv': (['url','label'], 'e0d305f8f994b39d6a26e8f5d6fc0d903431e486a9cd70358156fabd78b8eb10'),
          'legitphish.csv': (['URL','ClassLabel'], '81b5229a03b7f51739186c9ca614722238b2a74e32379b92401bff5bd0d65d3d')}
for f, (cols, h) in EXPECT.items():
    _d = pd.read_csv(f'{DATA_DIR}/{f}', usecols=cols, dtype=str, keep_default_na=False)
    got = chash(_d[cols[0]], _d[cols[1]]); print(f, 'content OK' if got == h else f'CONTENT DIFFERS {got[:12]} (results may differ)')
ph = pd.read_csv(f'{DATA_DIR}/phiusiil.csv', usecols=['URL','label'])
# label-coding guard: in this project 1 = phishing. The original PhiUSIIL release codes 1 = legitimate.
# Every PhiUSIIL legitimate URL has the form https://www.<host>, so the coding can be detected from the data.
_tpl = ph.URL.astype(str).str.strip().str.match(r'^https://www\.[a-z0-9.-]+$')
if _tpl[ph.label == 1].mean() > _tpl[ph.label == 0].mean():
    print('NOTE: phiusiil.csv uses the original coding (1 = legitimate) -> converted to project coding (1 = phishing)')
    ph['label'] = 1 - ph['label']
assert _tpl[ph.label == 0].mean() > 0.99, 'label coding check failed'
parts = np.load(f'{DATA_DIR}/source_partitions.npz')
up = pd.read_csv(f'{DATA_DIR}/urlphish.csv', usecols=['url','label'])
usel = np.load(f'{DATA_DIR}/urlphish_selection.npz')['corrected_full_row_indices']
lp = pd.read_csv(f'{DATA_DIR}/legitphish.csv', usecols=['URL','ClassLabel'])
lp = lp[lp.ClassLabel.notna()].reset_index(drop=True)       # drop the single unlabeled row
lsel = np.load(f'{DATA_DIR}/legitphish_selection.npz')['corrected_full_row_indices']
P = {}
for k, nm in [('source_train','train'),('source_validation','val'),('source_test','PhiUSIIL test')]:
    i = parts[k]; P[nm] = pd.DataFrame({'row_id':[f'PhiUSIIL:{j}' for j in i],'dataset':'PhiUSIIL',
                                        'url':ph.URL.values[i].astype(str),'label':ph.label.values[i].astype(int)})
P['URL-Phish'] = pd.DataFrame({'row_id':[f'URL-Phish:{j}' for j in usel],'dataset':'URL-Phish',
                               'url':up.url.values[usel].astype(str),'label':up.label.values[usel].astype(int)})
P['LegitPhish'] = pd.DataFrame({'row_id':[f'LegitPhish:{j}' for j in lsel],'dataset':'LegitPhish',
                                'url':lp.URL.values[lsel].astype(str),'label':(1-lp.ClassLabel.values[lsel]).astype(int)})
SIZES = {k: len(v) for k,v in P.items()}; print(SIZES)
assert SIZES == {'train':150931,'val':37716,'PhiUSIIL test':47148,'URL-Phish':116126,'LegitPhish':71767}
assert P['URL-Phish'].label.sum()==16600 and P['LegitPhish'].label.sum()==63678
SETS = ['PhiUSIIL test','URL-Phish','LegitPhish']
""")
code(r"""
# 4. URL variants — identical to code/common.py
SCH = re.compile(r'^([A-Za-z][A-Za-z0-9+.-]*)://')
def v_swap(u):
    u = u.strip(); m = SCH.match(u)
    if not m: return u
    s = m.group(1).lower()
    if s == 'https': return 'http://' + u[m.end():]
    if s == 'http': return 'https://' + u[m.end():]
    return u
def v_noscheme(u): return SCH.sub('', u.strip(), count=1)
def _split_host(rest):
    i = rest.find('/'); j = rest.find('?'); k = rest.find('#')
    cut = min([x for x in (i, j, k) if x >= 0], default=len(rest))
    return rest[:cut], rest[cut:]
def v_canonical(u):
    rest = v_noscheme(u); host, tail = _split_host(rest); host = host.lower()
    if host.startswith('www.'): host = host[4:]
    out = host + tail
    return out.rstrip('/') if out.rstrip('/') else out
def v_noslash(u):
    u = u.strip()
    return u.rstrip('/') if SCH.sub('', u).rstrip('/') else u
def v_template(u):
    rest = v_noscheme(u); host, tail = _split_host(rest); host = host.lower()
    if not host.startswith('www.') and not re.match(r'^\d{1,3}(\.\d{1,3}){3}(:\d+)?$', host): host = 'www.' + host
    out = 'https://' + host + tail
    return out.rstrip('/') if (host + tail).rstrip('/') else out
VARIANTS = {'swap':v_swap,'noscheme':v_noscheme,'canonical':v_canonical,'noslash':v_noslash,'template':v_template}
def v_hostonly(u):
    host, _ = _split_host(v_noscheme(u)); host = host.lower()
    return host[4:] if host.startswith('www.') else host
def url_parts(u):
    u = u.strip(); m = SCH.match(u); sch = m.group(1).lower() if m else None
    rest = u[m.end():] if m else u; host, tail = _split_host(rest)
    www = host.lower().startswith('www.')
    if www: host = host[4:]
    slash = (host + tail).endswith('/') and len((host + tail).rstrip('/')) > 0
    return sch, host, www, (tail.rstrip('/') if slash else tail), slash
def factorial_variant(u, scheme, www, slash, path):
    _, host, _, tail, _ = url_parts(u)
    return ('' if scheme == 'none' else scheme + '://') + ('www.' if www else '') + host + (tail if path else '') + ('/' if slash else '')
FACT_CELLS = [(sc, w, sl, p) for sc in ['https', 'http', 'none'] for w in (True, False) for sl in (True, False) for p in (True, False)]
assert factorial_variant('http://x.org/a/b/', 'https', True, False, False) == 'https://www.x.org'
INPUT_T = {'orig':lambda u:u.strip(),'swapaug':lambda u:u.strip(),'noscheme':v_noscheme,'canonical':v_canonical,'hostonly':v_hostonly}
NAMES = {'orig':f'{SHORT}','noscheme':f'{SHORT}-noscheme','swapaug':f'{SHORT}-swapaug','canonical':f'{SHORT}-canonical','hostonly':f'{SHORT}-hostonly'}
assert v_canonical('https://www.Abc.com/') == 'abc.com' and v_template('http://x.org/a/b/') == 'https://www.x.org/a/b'
""")
code(r"""
# 5. training / prediction
dev = torch.device('cuda'); tok = AutoTokenizer.from_pretrained(MODEL_NAME)
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
def batches(texts, bs, shuffle, rng=None):
    idx = np.argsort([len(t) for t in texts], kind='mergesort')
    ch = [idx[i:i+bs] for i in range(0, len(idx), bs)]
    if shuffle: rng.shuffle(ch)
    return ch
@torch.no_grad()
def predict(model, texts, bs=512):
    model.eval(); out = np.zeros(len(texts), np.float32)
    for ch in batches(texts, bs, False):
        b = tok([texts[i] for i in ch], truncation=True, max_length=MAX_LEN, padding=True, return_token_type_ids=False, return_tensors='pt').to(dev)
        with torch.autocast('cuda', dtype=AMP):
            out[ch] = model(**b).logits.float().squeeze(-1).cpu().numpy()
    return out
def run(mode, seed):
    name = NAMES[mode]; fn = f'{OUT_DIR}/transformer_{name}_s{seed}.csv.gz'
    if os.path.exists(fn): print('skip', fn); return
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); rng = np.random.default_rng(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(dev)
    T = INPUT_T[mode]; tr = [T(u) for u in P['train'].url]; ytr = P['train'].label.values.astype(np.float32)
    va = [T(u) for u in P['val'].url]; yva = P['val'].label.values
    steps = EPOCHS * int(np.ceil(len(tr) / BATCH))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler('cuda', enabled=(AMP == torch.float16)); lossf = nn.BCEWithLogitsLoss()
    best, best_state, log, t0 = np.inf, None, [], time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); te = time.time()
        cur = [v_swap(u) if f else u for u, f in zip(tr, rng.random(len(tr)) < .5)] if mode == 'swapaug' else tr
        tot = n = 0
        for ch in batches(cur, BATCH, True, rng):
            ch = rng.permutation(ch)
            b = tok([cur[i] for i in ch], truncation=True, max_length=MAX_LEN, padding=True, return_token_type_ids=False, return_tensors='pt').to(dev)
            y = torch.from_numpy(ytr[ch]).to(dev)
            with torch.autocast('cuda', dtype=AMP):
                z = model(**b).logits.float().squeeze(-1)
            loss = lossf(z, y); opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item() * len(ch); n += len(ch)
        zv = predict(model, va); vl = float(np.mean(np.logaddexp(0, zv) - yva * zv))
        log.append({'epoch': ep, 'train_loss': tot / n, 'val_logloss': vl, 'val_acc': float(np.mean((zv >= 0) == yva)), 'seconds': round(time.time() - te)})
        print(name, seed, log[-1], flush=True)
        if vl < best - 1e-4: best = vl; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: break   # early stopping, patience 1
    model.load_state_dict(best_state); rows = []
    for k in ['val'] + SETS:
        d = P[k]; variants = {'orig': d.url.values}
        if k != 'val':
            for vn, f in VARIANTS.items(): variants[vn] = np.array([f(u) for u in d.url.values], dtype=object)
        for vn, urls in variants.items():
            z = predict(model, [T(u) for u in urls])
            changed = np.zeros(len(d), bool) if vn == 'orig' else (urls != np.array([str(u).strip() for u in d.url], dtype=object))
            rows.append(pd.DataFrame({'row_id': d.row_id, 'dataset': d.dataset, 'partition': k, 'label': d.label, 'logit': z,
                                      'variant': vn, 'seed': seed, 'model': name, 'changed': changed}))
    pd.concat(rows).to_csv(fn, index=False, float_format='%.6f')
    json.dump({'model': name, 'base': MODEL_NAME, 'seed': seed, 'log': log, 'best_val_logloss': best,
               'seconds': round(time.time() - t0), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
               'transformers': __import__('transformers').__version__}, open(fn.replace('.csv.gz', '_train_log.json'), 'w'), indent=1)
    if mode == 'orig' and seed == 42: torch.save(best_state, f'{OUT_DIR}/transformer_{name}_s42.pt')
    if mode == 'orig':   # 24-cell factorial format edits on a fixed sample of URLs
        fr = []
        for k in SETS:
            d = P[k]
            for c in (0, 1):
                idx = d.index[d.label == c]
                idx = np.sort(np.random.default_rng(42).choice(idx, min(FACTORIAL_N, len(idx)), replace=False))
                urls = d.url.values[idx]
                for cell in FACT_CELLS:
                    z = predict(model, [factorial_variant(u, *cell) for u in urls])
                    cn = f"{cell[0]}|{'www' if cell[1] else 'nowww'}|{'slash' if cell[2] else 'noslash'}|{'path' if cell[3] else 'nopath'}"
                    fr.append(pd.DataFrame({'row_id': d.row_id.values[idx], 'partition': k, 'label': c, 'cell': cn, 'logit': z}))
        pd.concat(fr).assign(seed=seed, model=name).to_csv(f'{OUT_DIR}/transformer_factorial_{name}_s{seed}.csv.gz', index=False, float_format='%.5f')
    del model; torch.cuda.empty_cache()
for mode, seed in RUNS: run(mode, seed)
""")
code(r"""
# 6. Step 5 / E3: Integrated Gradients (PAD baseline, 50 steps) + occlusion on the ('orig', 42) model
from captum.attr import LayerIntegratedGradients
def scheme_end(u):
    m = SCH.match(u); return m.end() if m else 0     # span covers e.g. 'https://'
def spans(u):
    se = scheme_end(u)
    w = (se, se + 4) if u[se:se + 4].lower() == 'www.' else (0, 0)
    ts = (len(u) - 1, len(u)) if u.endswith('/') and len(u) > se + 1 else (0, 0)
    return {'scheme': (0, se), 'www': w, 'trailslash': ts}
def token_char_share(attr, offsets, span):
    # spread each token's |attribution| uniformly over its characters, then take the share inside the span
    s0, s1 = span
    if s1 <= s0: return np.nan
    tot = ins = 0.
    for a, (s, e) in zip(np.abs(attr), offsets):
        if e <= s: continue           # special tokens / padding
        tot += a; ins += a * max(0, min(e, s1) - max(s, s0)) / (e - s)
    return ins / tot if tot > 0 else np.nan
if DO_ATTRIBUTION:
    name = NAMES['orig']; afn = f'{OUT_DIR}/transformer_attr_{name}_s42.csv.gz'
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model.load_state_dict(torch.load(f'{OUT_DIR}/transformer_{name}_s42.pt')); model.to(dev).eval().float()
    pred = pd.read_csv(f'{OUT_DIR}/transformer_{name}_s42.csv.gz'); pred = pred[pred.variant == 'orig'].copy()
    pred['logit'] = MODEL_LOGIT_SIGN * pred['logit']     # phishing logit; labels come from P (project coding)
    emb_layer = model.get_input_embeddings()
    def fwd(ids, mask): return MODEL_LOGIT_SIGN * model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)   # phishing logit
    lig = LayerIntegratedGradients(fwd, emb_layer)
    out = []
    for s in ['URL-Phish', 'LegitPhish', 'PhiUSIIL test']:
        d = P[s].merge(pred[pred.partition == s][['row_id', 'logit']], on='row_id')
        leg = d[d.label == 0]
        for grp, sub in [('FP', leg[leg.logit >= 0]), ('TN', leg[leg.logit < 0])]:
            n_all = len(sub)
            sub = sub.sample(min(ATTR_N, len(sub)), random_state=42) if len(sub) else sub
            urls = [u[:MAX_LEN * 4] for u in sub.url.str.strip().tolist()]
            for i0 in range(0, len(urls), 32):
                uu = urls[i0:i0 + 32]
                b = tok(uu, truncation=True, max_length=MAX_LEN, padding=True, return_token_type_ids=False, return_tensors='pt', return_offsets_mapping=True)
                offs = b.pop('offset_mapping').numpy(); ids = b['input_ids'].to(dev); mask = b['attention_mask'].to(dev)
                base = ids.clone(); special = torch.isin(ids, torch.tensor(tok.all_special_ids, device=dev))
                base[~special] = tok.pad_token_id           # PAD baseline, keep [CLS]/[SEP]
                att = lig.attribute(ids, baselines=base, additional_forward_args=(mask,), n_steps=50, internal_batch_size=256)
                att = att.sum(-1).detach().cpu().numpy()
                # occlusion: replace each single token by PAD, delta logit
                with torch.no_grad():
                    z0 = fwd(ids, mask).float().cpu().numpy(); occ = np.zeros(ids.shape, np.float32)
                    for t in range(ids.shape[1]):
                        idt = ids.clone(); idt[:, t] = tok.pad_token_id
                        occ[:, t] = z0 - fwd(idt, mask).float().cpu().numpy()
                for j, u in enumerate(uu):
                    n = int(mask[j].sum()); sp = spans(u)
                    out.append({'set': s, 'group': grp, 'n_group_total': n_all, 'url': u, 'logit': float(z0[j]),
                                'ig_scheme': token_char_share(att[j][:n], offs[j][:n], sp['scheme']),
                                'occ_scheme': token_char_share(occ[j][:n], offs[j][:n], sp['scheme']),
                                'ig_www': token_char_share(att[j][:n], offs[j][:n], sp['www']),
                                'occ_www': token_char_share(occ[j][:n], offs[j][:n], sp['www']),
                                'ig_trailslash': token_char_share(att[j][:n], offs[j][:n], sp['trailslash']),
                                'occ_trailslash': token_char_share(occ[j][:n], offs[j][:n], sp['trailslash']),
                                'char_share_scheme': sp['scheme'][1] / len(u) if sp['scheme'][1] else np.nan,
                                'ig_completeness_gap': float(att[j].sum() - (z0[j] - fwd(base[j:j+1], mask[j:j+1]).item()))})
            print(s, grp, len(urls), flush=True)
    pd.DataFrame(out).to_csv(afn, index=False, float_format='%.6f')
    print(pd.DataFrame(out).groupby(['set', 'group'])[['ig_scheme', 'occ_scheme', 'ig_www', 'occ_www', 'ig_trailslash', 'occ_trailslash']].mean().round(3))
""")
code(r"""
# 7. zip outputs for upload back to the chat
import shutil
shutil.make_archive('/content/transformer_outputs' if IN_COLAB else '/kaggle/working/transformer_outputs', 'zip', OUT_DIR,)
print(sorted(os.listdir(OUT_DIR)))
""")

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1] if len(sys.argv) > 1 else 'transformer_gpu.ipynb', 'w'), indent=1)
# attribution-only notebook for the already-trained 28 Oct model (trained with 1 = legitimate coding)
import copy
nb2 = copy.deepcopy(nb)
nb2['cells'][0]['source'] = ('# Transformer attribution only (Step 5 / E3) - for the DistilBERT model you already trained\n'
    'Use the SAME Drive folders as before: `DATA_DIR` with the 6 input files and `OUT_DIR` that already holds '
    '`transformer_distilbert-base-cased_s42.pt` and `transformer_distilbert-base-cased_s42.csv.gz`.\n\n'
    'That model was trained with the original PhiUSIIL label coding (1 = legitimate), so `MODEL_LOGIT_SIGN = -1` is preset; '
    'it converts the output to the phishing logit. No training happens here (`RUNS = []`). '
    'Expected time on a T4: about 30-60 min. Upload `transformer_attr_distilbert-base-cased_s42.csv.gz` when done.')
c2 = nb2['cells'][2]['source']
c2 = re.sub(r"RUNS = \[.*?\]\nFACTORIAL_N", "RUNS = []   # no training in this notebook\nFACTORIAL_N", c2, flags=re.S)
c2 = c2.replace("MODEL_LOGIT_SIGN = 1     # set to -1 ONLY", "MODEL_LOGIT_SIGN = -1    # preset: -1 ONLY")
assert 'RUNS = []' in c2 and 'MODEL_LOGIT_SIGN = -1' in c2
nb2['cells'][2]['source'] = c2
json.dump(nb2, open((sys.argv[1] if len(sys.argv) > 1 else 'transformer_gpu.ipynb').replace('transformer_gpu', 'transformer_attr_only_gpu'), 'w'), indent=1)
print('cells', len(CELLS))
