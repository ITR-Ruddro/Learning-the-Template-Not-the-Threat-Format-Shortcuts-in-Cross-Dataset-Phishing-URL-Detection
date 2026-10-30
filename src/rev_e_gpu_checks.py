"""Revision E: checks on the GPU revision runs.
E1  label coding (#8): seed-42 DistilBERT retrained with the correct coding vs. the original sign-corrected seed-42 run
E2  DistilBERT factorial edits (10,000 URLs per set and class)
Outputs: results/rev_E1_label_coding_check.csv, results/rev_E2_factorial_distilbert.csv, results/rev_E2_residual_fp_by_path_distilbert.csv"""
import sys, numpy as np, pandas as pd
from common import *
P = load_all()
a = pd.read_csv(f'{ROOT}/predictions/transformer/DistilBERT_s42.csv.gz'); b = pd.read_csv(f'{ROOT}/predictions/transformer_check/DistilBERT_s42.csv.gz')
rows = []
for s in ['val'] + SETS:
    for v in (['orig'] if s == 'val' else ['orig'] + list(VARIANTS)):
        x = a[(a.partition == s) & (a.variant == v)].set_index('row_id'); y = b[(b.partition == s) & (b.variant == v)].set_index('row_id').loc[x.index]
        za, zb, lab = x.logit.values, y.logit.values, x.label.values
        r = {'partition': s, 'variant': v, 'n': len(za), 'pearson_logit': np.corrcoef(za, zb)[0, 1],
             'decision_agreement': np.mean((za >= 0) == (zb >= 0))}
        if s != 'val' and v == 'orig':
            ma, mb = metrics(za, lab), metrics(zb, lab)
            for k in ['F1', 'PR_AUC', 'FPR']: r[f'{k}_signcorrected'] = ma[k]; r[f'{k}_correct_coding'] = mb[k]
        rows.append(r)
pd.DataFrame(rows).to_csv(f'{RES}/rev_E1_label_coding_check.csv', index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))

F = pd.read_csv(sys.argv[1])  # transformer_factorial_distilbert-base-cased_s42.csv.gz
F['phish'] = F.logit >= 0
agg = F.groupby(['partition', 'label', 'cell']).agg(n=('row_id', 'size'), share_pred_phishing=('phish', 'mean'), mean_logit=('logit', 'mean')).reset_index()
agg['class'] = agg.label.map({0: 'legitimate', 1: 'phishing'}); agg = agg.rename(columns={'partition': 'set'}); agg['model'] = 'DistilBERT'
agg.to_csv(f'{RES}/rev_E2_factorial_distilbert.csv', index=False)
# residual FPs by path among the sampled URL-Phish legitimate URLs (original-input decision of the correct-coding run)
o = b[(b.partition == 'URL-Phish') & (b.variant == 'orig')].set_index('row_id').logit
url = P['URL-Phish'].set_index('row_id').url
x = F[(F.partition == 'URL-Phish') & (F.label == 0)].pivot(index='row_id', columns='cell', values='logit')
fp = o.reindex(x.index) >= 0; hp = pd.Series([len(url_parts(u)[3]) > 0 for u in url.reindex(x.index)], index=x.index)
rr = []
for g, m in [('has path', hp), ('no path', ~hp), ('all', hp | ~hp)]:
    k = fp & m
    rr.append({'group': g, 'n_FP_sampled': int(k.sum()), 'pct_corrected_template_path_kept': 100 * np.mean(x.loc[k, 'https|www|noslash|path'] < 0),
               'pct_corrected_template_host_only': 100 * np.mean(x.loc[k, 'https|www|noslash|nopath'] < 0)})
pd.DataFrame(rr).to_csv(f'{RES}/rev_E2_residual_fp_by_path_distilbert.csv', index=False)
print(pd.DataFrame(rr).round(2).to_string(index=False))
print(agg[(agg.set == 'URL-Phish')].pivot_table(index='cell', columns='class', values='share_pred_phishing').round(3).to_string())
