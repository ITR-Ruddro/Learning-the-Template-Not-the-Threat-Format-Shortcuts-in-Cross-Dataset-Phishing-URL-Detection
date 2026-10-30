"""Step 5 / E3: Integrated Gradients (Captum, PAD baseline, 50 steps) + single-character occlusion for the char-CNN (seed 42).
Legitimate URLs only: false positives (logit >= 0) vs true negatives, <= 5,000 each (random_state 42) per set.
Per URL we record the share of total |attribution| on: the scheme span ('http://' / 'https://'), the 'www.' span,
and a trailing '/', plus the span's share of characters (uniform-attribution reference).
Usage: python step5_attr_charcnn.py [model_name=charCNN] [seed=42]"""
import sys, time, numpy as np, pandas as pd, torch
from captum.attr import IntegratedGradients
sys.argv = [sys.argv[0], 'orig', '42'] + sys.argv[1:]
from step2_charcnn import CharCNN, encode, batchify, PAD
from common import *

NAME = sys.argv[3] if len(sys.argv) > 3 else 'charCNN'; SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 42
torch.set_num_threads(2)
model = CharCNN(); model.load_state_dict(torch.load(f'{ROOT}/models/{NAME}_s{SEED}.pt')); model.eval()
pred = pd.read_csv(f'{ROOT}/predictions/charcnn/{NAME}_s{SEED}.csv.gz'); pred = pred[pred.variant == 'orig']
P = load_all()
ig = IntegratedGradients(lambda e, ln: model.forward_emb(e, ln))


def spans(u):
    m = SCH.match(u); se = m.end() if m else 0
    host_start = se; w = (host_start, host_start + 4) if u[host_start:host_start + 4].lower() == 'www.' else (0, 0)
    ts = (len(u) - 1, len(u)) if u.endswith('/') and len(u) > se + 1 else (0, 0)
    return (0, se), w, ts


def share(a, sp):
    s, e = sp; tot = np.abs(a).sum()
    return np.abs(a[s:e]).sum() / tot if (e > s and tot > 0) else np.nan


out = []; t0 = time.time()
for s in ['URL-Phish', 'LegitPhish', 'PhiUSIIL test']:
    d = P[s].merge(pred[pred.partition == s][['row_id', 'logit']], on='row_id'); leg = d[d.label == 0]
    for grp, sub in [('FP', leg[leg.logit >= 0]), ('TN', leg[leg.logit < 0])]:
        n_all = len(sub)
        if len(sub) == 0: continue
        sub = sub.sample(min(5000, len(sub)), random_state=42)
        urls = [u.strip()[:200] for u in sub.url]
        order = np.argsort([len(u) for u in urls])
        for i0 in range(0, len(urls), 64):
            ix = order[i0:i0 + 64]; uu = [urls[i] for i in ix]
            x, ln = batchify(encode(uu))
            with torch.no_grad():
                e = model.emb(x); z0 = model.forward_emb(e, ln).numpy()
                zb = model.forward_emb(torch.zeros_like(e), ln).numpy()
            att, delta = ig.attribute(e, baselines=torch.zeros_like(e), additional_forward_args=(ln,), n_steps=50,
                                      internal_batch_size=64 * 50, return_convergence_delta=True)
            att = att.sum(-1).detach().numpy()
            # occlusion: each character replaced by PAD in turn
            L = x.shape[1]; occ = np.zeros(x.shape, np.float32)
            with torch.no_grad():
                for t in range(int(ln.max())):
                    xt = x.clone(); xt[:, t] = PAD
                    occ[:, t] = z0 - model(xt, ln).numpy()
            for j, u in enumerate(uu):
                n = len(u); a, o = att[j, :n], occ[j, :n]; sc, ww, ts = spans(u)
                out.append({'set': s, 'group': grp, 'n_group_total': n_all, 'url': u, 'logit': float(z0[j]),
                            'ig_scheme': share(a, sc), 'occ_scheme': share(o, sc), 'ig_www': share(a, ww), 'occ_www': share(o, ww),
                            'ig_trailslash': share(a, ts), 'occ_trailslash': share(o, ts),
                            'char_share_scheme': (sc[1] - sc[0]) / n if sc[1] > sc[0] else np.nan,
                            'ig_scheme_signed': float(a[sc[0]:sc[1]].sum()) if sc[1] > sc[0] else np.nan,
                            'ig_total_signed': float(a.sum()), 'logit_minus_baseline': float(z0[j] - zb[j]),
                            'ig_convergence_delta': float(delta[j])})
        print(s, grp, n_all, len(urls), round(time.time() - t0), flush=True)
df = pd.DataFrame(out)
df.to_csv(f'{RES}/step5_attr_{NAME}_s{SEED}.csv.gz', index=False, float_format='%.6f')
cols = ['ig_scheme', 'occ_scheme', 'ig_www', 'occ_www', 'ig_trailslash', 'occ_trailslash', 'char_share_scheme']
print(df.groupby(['set', 'group'])[cols].mean().round(3).to_string())
print('seconds', round(time.time() - t0))
