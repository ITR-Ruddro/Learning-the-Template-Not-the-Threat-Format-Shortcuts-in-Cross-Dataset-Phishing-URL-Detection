"""Tables I-IV of the paper, computed only from the CSV files in results/ (via paper_values.py, which logs every number).
Usage: python make_tables.py            -> prints the tables as Markdown and writes results/tables.md and results/number_sources.csv"""
import re, pandas as pd
from paper_values import *

# ----------------------------------------------------------------------------------- numbers
M = {'LightGBM': 'LightGBM', 'charCNN': 'char-CNN', 'DistilBERT': 'DistilBERT'}
SETS3 = ['PhiUSIIL test', 'URL-Phish', 'LegitPhish']
lgU = E1('LightGBM', 'URL-Phish', 'F1'); cnU = E1('charCNN', 'URL-Phish', 'F1'); dbU = E1('DistilBERT', 'URL-Phish', 'F1')
tplS = RULE('PhiUSIIL test', 'template', 'F1_phishing'); tplU = RULE('URL-Phish', 'template', 'F1_phishing'); tplUfpr = RULE('URL-Phish', 'template', 'FPR_legit')
tplL = RULE('LegitPhish', 'template', 'F1_phishing')

# ----------------------------------------------------------------------------------- tables (cells are inline-markup strings)
def table1():
    hdr = ['Set / class', 'n', 'https', 'http', 'www.', 'end /', 'path', 'IP', 'len', 'tmpl.']
    rows = []
    for ds, lab in [('PhiUSIIL', 'PhiUSIIL test'), ('URL-Phish', 'URL-Phish'), ('LegitPhish', 'LegitPhish')]:
        for cls, c in [('legitimate', 'leg.'), ('phishing', 'phi.')]:
            tm = RULE(lab, 'template', 'pct_legit_matching' if cls == 'legitimate' else 'pct_phish_matching')
            rows.append([f'{lab.replace(" test", "")} {c}', n(AUD(ds, cls, 'n')), f1(AUD(ds, cls, 'pct_https')), f1(AUD(ds, cls, 'pct_http')),
                         f1(AUD(ds, cls, 'pct_www')), f1(AUD(ds, cls, 'pct_trailing_slash')), f1(AUD(ds, cls, 'pct_path_beyond_root')),
                         f1(AUD(ds, cls, 'pct_ip_host')), f'{AUD(ds, cls, "median_length"):.0f}', f1(tm)])
    return dict(key='audit', wide=False, cols='l' + 'r' * 9, header=hdr, rows=rows, size='scriptsize',
                caption='Format audit of the evaluated URLs (% of rows; end / = trailing slash; len = median characters; tmpl. = % matching the source template defined in Section III).',
                note=f'Source: results/step0_scheme_audit.csv, results/step0_format_rule_baselines.csv. No URL in PhiUSIIL or URL-Phish lacks a scheme; {pp(AUD("LegitPhish", "phishing", "pct_no_scheme"))} of LegitPhish phishing URLs do.')


def table2():
    hdr = ['Model', 'Set', 'F1 [95% CI]', 'PR-AUC (sd)', 'FPR', 'TPR@val (FPR)', 'TPR@orc. (FPR)', 'Brier', 'ECE']
    rows = []
    for st in SETS3:
        for m in ['LightGBM', 'charCNN', 'DistilBERT']:
            g = lambda c: E1(m, st, c)
            rows.append([M[m], st.replace(' test', ''), f'{f3(g("F1"))} {ci(g("F1_lo"), g("F1_hi"))}', f'{f3(g("PR_AUC"))} ({f3(g("PR_AUC_seed_sd"))})',
                         f3(g('FPR')), f'{f3(g("TPR_at_valFPR1"))} ({f3(g("achFPR_valthr"))})', f'{f3(g("TPR_at_oracleFPR1"))} ({f3(g("achFPR_oraclethr"))})',
                         f3(g('Brier')), f3(g('ECE15'))])
    return dict(key='e1', wide=True, cols='llrrrrrrr', header=hdr, rows=rows, size='footnotesize', rule_every=3,
                caption='E1: Zero-shot transfer from PhiUSIIL (mean of seeds 42–44). F1 and FPR at p = 0.5; PR-AUC on the logit (sd = across seeds); TPR at FPR ≤ 1% with the threshold chosen on source validation (val) or on the target itself (orc., diagnostic), achieved FPR in parentheses; ECE with 15 bins. CIs: 1,000 class-stratified exact-URL bootstrap resamples.',
                note='Source: results/step3_E1_transfer.csv.')


def table3():
    hdr = ['Edit (URL-Phish legitimate)', 'LightGBM', 'char-CNN', 'DistilBERT']
    def cell(m, v, d='all', col='pct_errors_corrected'):
        return pc(CF(m, v, 'URL-Phish', 'legitimate', col, d), 1)
    rows = [['IsHTTPS feature 0↔1', cell('LightGBM', 'IsHTTPS_flip'), '–', '–'],
            ['http → https', cell('LightGBM', 'swap', 'http->https'), cell('charCNN', 'swap', 'http->https'), cell('DistilBERT', 'swap', 'http->https')],
            ['scheme removed', cell('LightGBM', 'noscheme'), cell('charCNN', 'noscheme'), cell('DistilBERT', 'noscheme')],
            ['trailing / removed', cell('LightGBM', 'noslash'), cell('charCNN', 'noslash'), cell('DistilBERT', 'noslash')],
            ['canonicalized', cell('LightGBM', 'canonical'), cell('charCNN', 'canonical'), cell('DistilBERT', 'canonical')],
            ['template projection', cell('LightGBM', 'template'), cell('charCNN', 'template'), cell('DistilBERT', 'template')],
            ['{i}Source legit., https → http: flip rate{/i}', pc(CF('LightGBM', 'swap', 'PhiUSIIL test', 'legitimate', 'flip_rate', 'https->http')),
             pc(CF('charCNN', 'swap', 'PhiUSIIL test', 'legitimate', 'flip_rate', 'https->http')), pc(CF('DistilBERT', 'swap', 'PhiUSIIL test', 'legitimate', 'flip_rate', 'https->http'))],
            ['{i}… mean Δlogit{/i}', sgn(CF('LightGBM', 'swap', 'PhiUSIIL test', 'legitimate', 'mean_dlogit', 'https->http'), 1),
             sgn(CF('charCNN', 'swap', 'PhiUSIIL test', 'legitimate', 'mean_dlogit', 'https->http'), 1), sgn(CF('DistilBERT', 'swap', 'PhiUSIIL test', 'legitimate', 'mean_dlogit', 'https->http'), 1)]]
    return dict(key='e2', wide=False, cols='lrrr', header=hdr, rows=rows, size='footnotesize', rule_before=[6],
                caption='E2: Share of URL-Phish false positives (legitimate URLs scored as phishing) that become correct under each edit (mean of seeds 42–44). Bottom rows: effect of turning https into http on PhiUSIIL legitimate test URLs.',
                note='Source: results/step4_E2_counterfactual.csv (CIs, flip rates, Δlogit and both classes are in the file).')


def table4():
    hdr = ['Training', 'ΔF1 source', 'URL-Phish F1', 'URL-Phish PR-AUC', 'URL-Phish FPR', 'ΔF1 LegitPh.']
    rows = []
    for fam, lab in [('charCNN', 'char-CNN'), ('DistilBERT', 'DistilBERT')]:
        for v, vl in [('noscheme', 'scheme stripped'), ('swapaug', 'random swap'), ('canonical', 'canonicalized')]:
            g = lambda st, c: E4(fam, v, st, c)
            rows.append([f'{lab}: {vl}', f'{sgn(g("PhiUSIIL test", "dF1"))} {ci(g("PhiUSIIL test", "dF1_lo"), g("PhiUSIIL test", "dF1_hi"))}',
                         f'{f3(g("URL-Phish", "F1_a"))}→{f3(g("URL-Phish", "F1_b"))}', f'{f3(g("URL-Phish", "PR_AUC_a"))}→{f3(g("URL-Phish", "PR_AUC_b"))}',
                         f'{f3(g("URL-Phish", "FPR_a"))}→{f3(g("URL-Phish", "FPR_b"))}', sgn(g('LegitPhish', 'dF1'))])
    return dict(key='e4', wide=True, cols='lrrrrr', header=hdr, rows=rows, size='footnotesize', rule_every=3,
                caption='E4: Scheme- and format-neutral training versus the unmodified model of the same family (Δ = variant − unmodified; paired bootstrap over the same 1,000 resamples). char-CNN: mean of seeds 42–44; DistilBERT: seed 42 only. All target ΔF1, ΔPR-AUC and ΔFPR CIs exclude zero.',
                note='Source: results/step6_E4_charCNN_paired_delta.csv, results/step6_E4_DistilBERT_paired_delta.csv.')




def md(T, title):
    clean = lambda s: re.sub(r'\{/?i\}', '*', s)
    out = [f'### {title}', '', clean(T['caption']), '', '| ' + ' | '.join(T['header']) + ' |', '|' + '---|' * len(T['header'])]
    out += ['| ' + ' | '.join(clean(c) for c in r) + ' |' for r in T['rows']]
    return '\n'.join(out + ['', T['note'], ''])


if __name__ == '__main__':
    txt = '\n'.join([md(table1(), 'Table I'), md(table2(), 'Table II'), md(table3(), 'Table III'), md(table4(), 'Table IV')])
    open(f'{RES}/tables.md', 'w').write(txt)
    pd.DataFrame(LOG).drop_duplicates().to_csv(f'{RES}/number_sources.csv', index=False)
    print(txt)
