"""Fig. 2: transfer of the format-only rule between five datasets (MCC); diagonal = in-dataset MCC.
Input: results/rev_F_audit_5datasets.csv, results/rev_F_format_rule_transfer.csv -> figures/fig2_format_transfer.pdf/.png"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from common import RES, ROOT

A = pd.read_csv(f'{RES}/rev_F_audit_5datasets.csv'); T = pd.read_csv(f'{RES}/rev_F_format_rule_transfer.csv')
names = ['PhiUSIIL', 'LegitPhish', 'URL-Phish', 'Sahingoz2019', 'MaliciousURL-GH']
labels = ['PhiUSIIL', 'LegitPhish', 'URL-Phish', 'Sahingoz', 'Malicious-GH']
M = np.zeros((5, 5))
for i, a in enumerate(names):
    for j, b in enumerate(names):
        M[i, j] = A.set_index('dataset').loc[a, 'format_rule_MCC'] if a == b else T[(T.train == a) & (T.test == b)].MCC.iloc[0]
plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
cmap = LinearSegmentedColormap.from_list('div', ['#eb6834', '#f2f2f2', '#2a78d6'])
fig, ax = plt.subplots(figsize=(3.4, 2.15))
im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.5, color='#111111', fontweight='bold' if i == j else 'normal')
ax.set_xticks(range(5)); ax.set_xticklabels(labels, rotation=25, ha='right'); ax.set_yticks(range(5)); ax.set_yticklabels(labels)
ax.set_xlabel('evaluated on'); ax.set_ylabel('format rule fitted on')
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5, 5), minor=True); ax.set_yticks(np.arange(-.5, 5), minor=True)
ax.grid(which='minor', color='white', lw=1.5); ax.tick_params(which='both', length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[-1, 0, 1]); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6, length=0)
cb.set_label('MCC', fontsize=6.5)
plt.tight_layout(pad=0.3)
fig.savefig(f'{ROOT}/figures/fig2_format_transfer.pdf'); fig.savefig(f'{ROOT}/figures/fig2_format_transfer.png', dpi=300)
print(np.round(M, 3))
