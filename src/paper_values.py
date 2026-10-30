"""Single source of every number printed in the paper. Each accessor logs (key, value, source file) so that
paper/number_sources.csv lists the provenance of every number used."""
import os, pandas as pd
from common import RES

LOG = []
_cache = {}


def _csv(name):
    if name not in _cache:
        _cache[name] = pd.read_csv(os.path.join(RES, name))
    return _cache[name]


def _get(name, mask_fn, col, key):
    d = _csv(name); r = d[mask_fn(d)]
    assert len(r) == 1, (name, key, len(r))
    v = float(r[col].iloc[0]); LOG.append({'key': key, 'column': col, 'value': v, 'file': f'results/{name}'}); return v


def AUD(ds, cls, col, cohort=None):
    cohort = cohort or {'PhiUSIIL': 'test', 'URL-Phish': 'evaluated cohort', 'LegitPhish': 'evaluated cohort'}[ds]
    return _get('step0_scheme_audit.csv', lambda d: (d.dataset == ds) & (d.cohort == cohort) & (d['class'] == cls), col, f'audit/{ds}/{cohort}/{cls}')


def RULE(st, rule, col):
    st = {'PhiUSIIL test': 'PhiUSIIL test', 'URL-Phish': 'URL-Phish cohort', 'LegitPhish': 'LegitPhish cohort'}[st]
    return _get('step0_format_rule_baselines.csv', lambda d: (d.set == st) & d.rule.str.startswith(rule), col, f'rule/{st}/{rule}')


def E1(model, st, col):
    return _get('step3_E1_transfer.csv', lambda d: (d.model == model) & (d.set == st), col, f'E1/{model}/{st}')


def EB(model, st, col):
    return _get('step3_E1_error_breakdown.csv', lambda d: (d.model == model) & (d.set == st), col, f'E1err/{model}/{st}')


def CF(model, variant, st, cls, col, direction='all'):
    return _get('step4_E2_counterfactual.csv', lambda d: (d.model == model) & (d.variant == variant) & (d.set == st) & (d.cls == cls) & (d.direction == direction),
                col, f'E2/{model}/{variant}/{direction}/{st}/{cls}')


def E3(model, st, group, measure, col='mean'):
    return _get('step5_E3_summary.csv', lambda d: (d.model == model) & (d.set == st) & (d.group == group) & (d.measure == measure), col, f'E3/{model}/{st}/{group}/{measure}')


def SG(st, group, scheme, col):
    return _get('step5_E3_signed_charCNN.csv', lambda d: (d.set == st) & (d.group == group) & (d.scheme == scheme), col, f'E3signed/{st}/{group}/{scheme}')


def E4(fam, variant, st, col):
    return _get(f'step6_E4_{fam}_paired_delta.csv', lambda d: (d.b == f'{fam}-{variant}/orig') & (d.set == st), col, f'E4/{fam}/{variant}/{st}')


def E4P(fam, variant, st, col):
    return _get(f'step6_E4_{fam}_performance.csv', lambda d: (d.model == f'{fam}-{variant}') & (d.set == st), col, f'E4perf/{fam}/{variant}/{st}')


def L1(model, st, col):
    return _get('step1_lgbm_IsHTTPS_flip.csv', lambda d: (d.set == st) & (d.cls == 'legitimate') & (d.direction == 'both'), col, f'L1/{st}')


def IGQ(model, col):
    return _get('step5_E3_ig_quality.csv', lambda d: d.model == model, col, f'IGquality/{model}')


def E4MAXABS(fam_list, st, col):
    vals = []
    for fam in fam_list:
        d = _csv(f'step6_E4_{fam}_paired_delta.csv'); vals += list(d[d.set == st][col].abs())
    v = max(vals); LOG.append({'key': f'E4maxabs/{st}', 'column': col, 'value': v, 'file': 'results/step6_E4_*_paired_delta.csv'}); return v


# formatting
def f3(x): return f'{x:.3f}'
def f2(x): return f'{x:.2f}'
def f1(x): return f'{x:.1f}'
def pc(x, d=1): return f'{100 * x:.{d}f}%'          # fraction -> percent
def pp(x, d=1): return f'{x:.{d}f}%'                # already percent
def sgn(x, d=3): return ('+' if x >= 0 else '−') + f'{abs(x):.{d}f}'
def ci(lo, hi, d=3): return f'[{lo:.{d}f}, {hi:.{d}f}]'
def n(x): return f'{int(round(x)):,}'
