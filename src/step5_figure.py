"""Step 5 / E3 summary + Fig. 1.
Inputs: results/step5_attr_charCNN_s42.csv.gz (CPU, this repo), results/step5_attr_DistilBERT_s42.csv.gz (author's GPU run).
Outputs: results/step5_E3_summary.csv (mean share per span with 95% bootstrap CI over URLs, 1,000 resamples, seed 7),
         figures/fig1_attribution.pdf / .png"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

rng = np.random.default_rng(7)
SRC = {'char-CNN': f'{RES}/step5_attr_charCNN_s42.csv.gz', 'DistilBERT': f'{RES}/step5_attr_DistilBERT_s42.csv.gz'}
rows = []
for m, f in SRC.items():
    A = pd.read_csv(f)
    for (s, g), d in A.groupby(['set', 'group']):
        for col in ['ig_scheme', 'occ_scheme', 'ig_www', 'occ_www', 'ig_trailslash', 'occ_trailslash', 'char_share_scheme']:
            x = d[col].dropna().values
            if len(x) == 0:
                rows.append(dict(model=m, set=s, group=g, measure=col, n=0)); continue
            bs = [rng.choice(x, len(x)).mean() for _ in range(1000)]
            rows.append(dict(model=m, set=s, group=g, measure=col, n=len(x), n_group_total=int(d.n_group_total.iloc[0]),
                             mean=x.mean(), median=np.median(x), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
S = pd.DataFrame(rows); S.to_csv(f'{RES}/step5_E3_summary.csv', index=False)
A = pd.read_csv(SRC['char-CNN']); A['scheme'] = np.where(A.url.str.lower().str.startswith('https://'), 'https', np.where(A.url.str.lower().str.startswith('http://'), 'http', 'none'))
sg = []
for keys, d in list(A.groupby(['set', 'group'])) + [((s, g, sc), d) for (s, g, sc), d in A.groupby(['set', 'group', 'scheme'])]:
    x = d.ig_scheme_signed.values; bs = [rng.choice(x, len(x)).mean() for _ in range(1000)]
    sg.append(dict(model='char-CNN', set=keys[0], group=keys[1], scheme=keys[2] if len(keys) > 2 else 'all', n=len(x),
                   mean_signed_IG_scheme=x.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), pct_toward_phishing=float((x > 0).mean())))
pd.DataFrame(sg).to_csv(f'{RES}/step5_E3_signed_charCNN.csv', index=False)

# ---------------------------------------------------------------- figure
plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42})
C = {'FP': '#eb6834', 'TN': '#2a78d6'}
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True)
ylab = []
for ax, m in zip(axes, SRC):
    y = 0; ticks = []
    for s in ['URL-Phish', 'LegitPhish']:
        for g in ['FP', 'TN']:
            for meas, mk, off in [('ig_scheme', 'o', 0.13), ('occ_scheme', 'D', -0.13)]:
                r = S[(S.model == m) & (S.set == s) & (S.group == g) & (S.measure == meas)].iloc[0]
                face = C[g] if meas == 'ig_scheme' else 'white'
                ax.errorbar(r['mean'], y + off, xerr=[[r['mean'] - r['lo']], [r['hi'] - r['mean']]], fmt=mk, ms=3.6,
                            mfc=face, mec=C[g], ecolor=C[g], elinewidth=0.8, capsize=0, mew=0.9, zorder=3)
            cs = S[(S.model == m) & (S.set == s) & (S.group == g) & (S.measure == 'char_share_scheme')].iloc[0]['mean']
            ax.plot([cs, cs], [y - 0.32, y + 0.32], color='#777777', lw=0.9, zorder=2)
            n = int(S[(S.model == m) & (S.set == s) & (S.group == g) & (S.measure == 'ig_scheme')].iloc[0]['n'])
            ticks.append((y, f'{s} {g}'))
            y -= 1
        y -= 0.4
    ax.set_xlim(0, 1); ax.set_title(m, fontsize=8)
    ax.grid(axis='x', color='#dddddd', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    ax.set_yticks([t[0] for t in ticks]); ax.set_yticklabels([t[1] for t in ticks], fontsize=6.5)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(['0', '.25', '.5', '.75', '1'])
    ax.tick_params(axis='y', length=0)
fig.text(0.6, 0.015, 'share of |attribution| on the scheme span', ha='center', fontsize=7)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='', color='#444444', ms=3.6, label='Integrated Gradients'),
     Line2D([], [], marker='D', ls='', mfc='white', mec='#444444', ms=3.6, label='Occlusion'),
     Line2D([], [], color='#777777', lw=0.9, label='char. share')]
fig.legend(handles=h, loc='upper center', ncol=3, frameon=False, fontsize=6.3, bbox_to_anchor=(0.58, 1.01), handletextpad=0.3, columnspacing=0.8)
plt.tight_layout(rect=(0, 0.04, 1, 0.92), w_pad=0.6)
fig.savefig(f'{ROOT}/figures/fig1_attribution.pdf'); fig.savefig(f'{ROOT}/figures/fig1_attribution.png', dpi=300)
pd.set_option('display.width', 250)
print(S[S.measure.isin(['ig_scheme', 'occ_scheme', 'char_share_scheme', 'ig_www', 'occ_www', 'ig_trailslash'])].pivot_table(
    index=['model', 'set', 'group'], columns='measure', values='mean').round(3).to_string())
