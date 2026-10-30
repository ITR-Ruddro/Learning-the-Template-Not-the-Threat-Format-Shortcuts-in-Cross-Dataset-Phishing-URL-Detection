"""Step 2a / 6: URLNet-style character CNN on raw URL text (CPU).
Usage: python step2_charcnn.py <train_mode> <seed>
train_mode: orig | noscheme (scheme stripped from every input) | swapaug (random http<->https swap, p=0.5, training only)
            | canonical (format-neutral: scheme, www., trailing '/' stripped, host lowercased, every input)
Architecture: char embedding 32 -> Conv1d 256 filters each of width 3/4/5/6 + ReLU -> global max-pool (masked to real
positions) -> dropout 0.5 -> dense 512 + ReLU -> dropout 0.5 -> 1 logit. Max length 200. Adam 1e-3, batch 256,
<= 6 epochs, early stopping on validation log-loss (patience 1), best epoch restored.
Saves predictions/charcnn/<model>_s<seed>.csv.gz with logits for val + 3 evaluation sets, under the original inputs and
the five E2 counterfactual variants (each passed through the model's own input transform)."""
import sys, time, json, random, numpy as np, pandas as pd, torch, torch.nn as nn
from common import *

MODE, SEED = sys.argv[1], int(sys.argv[2])
MAX_EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
OUTSUB = sys.argv[4] if len(sys.argv) > 4 else 'charcnn'
NAME = {'orig': 'charCNN', 'noscheme': 'charCNN-noscheme', 'swapaug': 'charCNN-swapaug', 'canonical': 'charCNN-canonical', 'hostonly': 'charCNN-hostonly'}[MODE]
torch.set_num_threads(2); torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
MAXLEN, PAD, UNK, EXTRA = 200, 0, 1, 5


def enc_char(c):
    o = ord(c)
    return o - 30 if 32 <= o <= 126 else UNK  # 2..96


V = 97
INPUT_T = {'orig': lambda u: u.strip(), 'swapaug': lambda u: u.strip(), 'noscheme': v_noscheme, 'canonical': v_canonical, 'hostonly': v_hostonly}[MODE]


def encode(urls):
    return [np.array([enc_char(c) for c in u[:MAXLEN]] or [UNK], dtype=np.int64) for u in urls]


def batchify(seqs):
    L = max(len(s) for s in seqs) + EXTRA
    x = np.zeros((len(seqs), L), np.int64); ln = np.zeros(len(seqs), np.int64)
    for i, s in enumerate(seqs): x[i, :len(s)] = s; ln[i] = len(s)
    return torch.from_numpy(x), torch.from_numpy(ln)


class CharCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 32, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(32, 256, k) for k in (3, 4, 5, 6)])
        self.drop = nn.Dropout(0.5); self.fc = nn.Linear(1024, 512); self.out = nn.Linear(512, 1)

    def forward_emb(self, e, ln):  # e: B x L x 32 (exposed for Integrated Gradients)
        h = e.transpose(1, 2); pooled = []
        for conv in self.convs:
            c = torch.relu(conv(h))  # B x 256 x (L-k+1)
            pos = torch.arange(c.shape[2], device=c.device)[None, None, :]
            c = c.masked_fill(pos >= ln[:, None, None], float('-inf'))  # windows must start inside the URL
            pooled.append(c.max(2).values)
        z = self.drop(torch.cat(pooled, 1))
        return self.out(self.drop(torch.relu(self.fc(z)))).squeeze(1)

    def forward(self, x, ln):
        return self.forward_emb(self.emb(x), ln)


def batches_by_length(seqs, bs, shuffle, rng=None):
    idx = np.argsort([len(s) for s in seqs], kind='mergesort')
    chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle: rng.shuffle(chunks)
    return chunks


@torch.no_grad()
def predict(model, urls, bs=1024):
    model.eval(); seqs = encode([INPUT_T(u) for u in urls]); out = np.zeros(len(seqs), np.float32)
    for ch in batches_by_length(seqs, bs, False):
        x, ln = batchify([seqs[i] for i in ch]); out[ch] = model(x, ln).numpy()
    return out


if __name__ == '__main__':
    t0 = time.time(); P = load_all(); rng = np.random.default_rng(SEED)
    tr_urls = P['train'].url.values; ytr = P['train'].label.values.astype(np.float32)
    va_urls = P['val'].url.values; yva = P['val'].label.values
    model = CharCNN(); opt = torch.optim.Adam(model.parameters(), 1e-3); lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad, log = np.inf, None, 0, []
    base_tr = [INPUT_T(u) for u in tr_urls]
    for ep in range(1, MAX_EPOCHS + 1):
        te = time.time(); model.train()
        if MODE == 'swapaug':  # fresh random swap each epoch
            flip = rng.random(len(tr_urls)) < 0.5
            cur = [v_swap(u) if f else u for u, f in zip(base_tr, flip)]
        else:
            cur = base_tr
        seqs = encode(cur); tot, n = 0., 0
        for ch in batches_by_length(seqs, 256, True, rng):
            ch = rng.permutation(ch)
            x, ln = batchify([seqs[i] for i in ch]); y = torch.from_numpy(ytr[ch])
            opt.zero_grad(); loss = lossf(model(x, ln), y); loss.backward(); opt.step()
            tot += loss.item() * len(ch); n += len(ch)
        zv = predict(model, va_urls)
        vl = float(np.mean(np.logaddexp(0, zv) - yva * zv)); vacc = float(np.mean((zv >= 0) == yva))
        log.append({'epoch': ep, 'train_loss': tot / n, 'val_logloss': vl, 'val_acc': vacc, 'seconds': round(time.time() - te, 1)})
        print(log[-1], flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 1: break
    model.load_state_dict(best_state)
    MD = 'models' if OUTSUB == 'charcnn' else f'models_{OUTSUB}'
    os.makedirs(f'{ROOT}/{MD}', exist_ok=True); os.makedirs(f'{ROOT}/predictions/{OUTSUB}', exist_ok=True)
    torch.save(best_state, f'{ROOT}/{MD}/{NAME}_s{SEED}.pt')
    rows = []; ti = time.time()
    for k in ['val'] + SETS:
        d = P[k]
        variants = {'orig': d.url.values}
        if k != 'val':
            for vn, f in VARIANTS.items(): variants[vn] = np.array([f(u) for u in d.url.values], dtype=object)
        for vn, urls in variants.items():
            z = predict(model, urls)
            changed = np.ones(len(d), bool) if vn == 'orig' else (urls != np.array([str(u).strip() for u in d.url], dtype=object))
            rows.append(pd.DataFrame({'row_id': d.row_id, 'dataset': d.dataset, 'partition': k, 'label': d.label, 'logit': z,
                                      'variant': vn, 'seed': SEED, 'model': NAME, 'changed': changed if vn != 'orig' else False}))
    pd.concat(rows).to_csv(f'{ROOT}/predictions/{OUTSUB}/{NAME}_s{SEED}.csv.gz', index=False, float_format='%.6f')
    json.dump({'model': NAME, 'seed': SEED, 'log': log, 'best_val_logloss': best, 'best_epoch': int(np.argmin([l['val_logloss'] for l in log]) + 1),
               'n_params': sum(p.numel() for p in model.parameters()), 'train_seconds': round(ti - t0), 'infer_seconds': round(time.time() - ti),
               'torch': torch.__version__, 'max_epochs': MAX_EPOCHS}, open(f'{ROOT}/predictions/{OUTSUB}/{NAME}_s{SEED}.json', 'w'), indent=1)
    print('done', NAME, SEED, round(time.time() - t0), 's')
