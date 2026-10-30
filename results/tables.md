### Table I

Format audit of the evaluated URLs (% of rows; end / = trailing slash; len = median characters; tmpl. = % matching the source template defined in Section III).

| Set / class | n | https | http | www. | end / | path | IP | len | tmpl. |
|---|---|---|---|---|---|---|---|---|---|
| PhiUSIIL leg. | 26,987 | 100.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 27 | 100.0 |
| PhiUSIIL phi. | 20,161 | 48.7 | 51.3 | 41.3 | 36.4 | 28.4 | 0.6 | 34 | 0.6 |
| URL-Phish leg. | 99,526 | 34.8 | 65.2 | 76.0 | 74.8 | 24.1 | 0.0 | 26 | 4.2 |
| URL-Phish phi. | 16,600 | 91.4 | 8.6 | 2.5 | 40.5 | 58.0 | 0.0 | 38 | 0.1 |
| LegitPhish leg. | 8,089 | 99.9 | 0.1 | 95.8 | 2.5 | 6.6 | 0.0 | 27 | 92.5 |
| LegitPhish phi. | 63,678 | 5.5 | 92.4 | 1.0 | 1.4 | 99.4 | 77.0 | 34 | 0.0 |

Source: results/step0_scheme_audit.csv, results/step0_format_rule_baselines.csv. No URL in PhiUSIIL or URL-Phish lacks a scheme; 2.0% of LegitPhish phishing URLs do.

### Table II

E1: Zero-shot transfer from PhiUSIIL (mean of seeds 42–44). F1 and FPR at p = 0.5; PR-AUC on the logit (sd = across seeds); TPR at FPR ≤ 1% with the threshold chosen on source validation (val) or on the target itself (orc., diagnostic), achieved FPR in parentheses; ECE with 15 bins. CIs: 1,000 class-stratified exact-URL bootstrap resamples.

| Model | Set | F1 [95% CI] | PR-AUC (sd) | FPR | TPR@val (FPR) | TPR@orc. (FPR) | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| LightGBM | PhiUSIIL | 0.984 [0.983, 0.985] | 0.998 (0.000) | 0.007 | 0.984 (0.010) | 0.984 (0.010) | 0.010 | 0.015 |
| char-CNN | PhiUSIIL | 0.998 [0.997, 0.998] | 0.999 (0.000) | 0.000 | 0.997 (0.010) | 0.997 (0.010) | 0.002 | 0.002 |
| DistilBERT | PhiUSIIL | 0.998 [0.998, 0.998] | 0.999 (0.000) | 0.000 | 0.997 (0.010) | 0.997 (0.010) | 0.002 | 0.001 |
| LightGBM | URL-Phish | 0.269 [0.268, 0.271] | 0.229 (0.036) | 0.782 | 0.908 (0.793) | 0.043 (0.010) | 0.661 | 0.664 |
| char-CNN | URL-Phish | 0.258 [0.258, 0.258] | 0.144 (0.015) | 0.957 | 0.999 (0.958) | 0.017 (0.010) | 0.820 | 0.820 |
| DistilBERT | URL-Phish | 0.258 [0.258, 0.258] | 0.379 (0.203) | 0.957 | 0.999 (0.958) | 0.007 (0.000) | 0.821 | 0.821 |
| LightGBM | LegitPhish | 0.995 [0.995, 0.996] | 1.000 (0.000) | 0.073 | 1.000 (0.076) | 0.939 (0.009) | 0.008 | 0.007 |
| char-CNN | LegitPhish | 0.995 [0.995, 0.996] | 1.000 (0.000) | 0.075 | 1.000 (0.083) | 0.967 (0.010) | 0.008 | 0.009 |
| DistilBERT | LegitPhish | 0.995 [0.995, 0.996] | 0.997 (0.000) | 0.075 | 1.000 (0.083) | 0.197 (0.003) | 0.008 | 0.009 |

Source: results/step3_E1_transfer.csv.

### Table III

E2: Share of URL-Phish false positives (legitimate URLs scored as phishing) that become correct under each edit (mean of seeds 42–44). Bottom rows: effect of turning https into http on PhiUSIIL legitimate test URLs.

| Edit (URL-Phish legitimate) | LightGBM | char-CNN | DistilBERT |
|---|---|---|---|
| IsHTTPS feature 0↔1 | 57.5% | – | – |
| http → https | 67.7% | 3.6% | 3.6% |
| scheme removed | 0.0% | 3.5% | 6.0% |
| trailing / removed | 1.4% | 18.9% | 18.8% |
| canonicalized | 0.0% | 32.6% | 59.0% |
| template projection | 83.3% | 78.0% | 77.9% |
| *Source legit., https → http: flip rate* | 99.3% | 100.0% | 100.0% |
| *… mean Δlogit* | +8.2 | +24.2 | +18.6 |

Source: results/step4_E2_counterfactual.csv (CIs, flip rates, Δlogit and both classes are in the file).

### Table IV

E4: Scheme- and format-neutral training versus the unmodified model of the same family (Δ = variant − unmodified; paired bootstrap over the same 1,000 resamples). char-CNN: mean of seeds 42–44; DistilBERT: seed 42 only. All target ΔF1, ΔPR-AUC and ΔFPR CIs exclude zero.

| Training | ΔF1 source | URL-Phish F1 | URL-Phish PR-AUC | URL-Phish FPR | ΔF1 LegitPh. |
|---|---|---|---|---|---|
| char-CNN: scheme stripped | −0.113 [-0.116, -0.109] | 0.258→0.264 | 0.144→0.591 | 0.957→0.927 | −0.001 |
| char-CNN: random swap | −0.115 [-0.118, -0.111] | 0.258→0.263 | 0.144→0.746 | 0.957→0.935 | −0.001 |
| char-CNN: canonicalized | −0.133 [-0.137, -0.129] | 0.258→0.542 | 0.144→0.521 | 0.957→0.268 | −0.001 |
| DistilBERT: scheme stripped | −0.103 [-0.106, -0.099] | 0.258→0.267 | 0.175→0.760 | 0.957→0.914 | −0.001 |
| DistilBERT: random swap | −0.106 [-0.109, -0.103] | 0.258→0.264 | 0.175→0.710 | 0.957→0.929 | −0.001 |
| DistilBERT: canonicalized | −0.113 [-0.116, -0.110] | 0.258→0.546 | 0.175→0.550 | 0.957→0.268 | −0.001 |

Source: results/step6_E4_charCNN_paired_delta.csv, results/step6_E4_DistilBERT_paired_delta.csv.
