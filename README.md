# Raw-URL text models and format shortcuts in cross-dataset phishing detection

Code, split indices and result files for the paper

> **Learning the Template, Not the Threat: Raw-URL Text Models Inherit Format Shortcuts in Cross-Dataset Phishing Detection**
> [Authors], [Conference], 2026.

The study trains a LightGBM reference, a URLNet-style character CNN, and DistilBERT on **PhiUSIIL**. It evaluates them zero-shot on **URL-Phish** and **LegitPhish**, and asks four questions:
- whether the text models rely on the URL scheme (counterfactual URL edits);
- what the models attend to (Integrated Gradients and occlusion);
- whether the failure sits in the legitimate class;
- whether scheme-neutral or format-neutral training helps.

Every number in the paper comes from a CSV in `results/`. `src/make_tables.py` rebuilds Tables I–IV from those files and lists the source of every number (`results/number_sources.csv`).

## Repository layout

| Path | Content |
|---|---|
| `src/prepare_data.py` | turns the three original dataset downloads into `data/phiusiil.csv`, `data/urlphish.csv`, `data/legitphish.csv` and checks their content hashes |
| `src/common.py` | data loading with the saved split/cohort indices, the five URL edits, the 15 lexical features, metrics, grouped bootstrap |
| `src/evaluate.py` | performance tables, paired deltas, counterfactual tables |
| `src/step0_*.py` … `src/step5_*.py` | one script per step (see below) |
| `src/import_transformer.py` | checks and imports the per-row logits written by the GPU notebook |
| `src/make_tables.py`, `src/paper_values.py` | Tables I–IV from `results/` + number provenance |
| `notebooks/transformer_gpu.ipynb` | DistilBERT training, counterfactual predictions and attribution (Colab/Kaggle GPU) |
| `splits/` | saved indices: PhiUSIIL train/validation/test (`source_partitions.npz`) and the evaluated target cohorts |
| `results/` | all result files used in the paper |
| `figures/` | Fig. 1 |
| `scripts/run_all.sh` | runs every CPU step in order |

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11
pip install -r requirements.txt
```

## 2. Get the data (not redistributed here)

Download the original files from Mendeley Data:
- **PhiUSIIL Phishing URL Dataset**: https://data.mendeley.com/datasets/shwpxscxy2 (Prasad & Chandra, *Computers & Security*, 2024)
- **URL-Phish**: https://data.mendeley.com/datasets/65z9twcx3r (version 2, `URL-Phish.csv`, 116,600 rows)
- **LegitPhish Dataset**: https://data.mendeley.com/datasets/hx4m73v2sf (Potpelwar et al., *Data in Brief*, 2025; 101,219 rows incl. one unlabeled row)

Then run:

```bash
python src/prepare_data.py --phiusiil PhiUSIIL_Phishing_URL_Dataset.csv \
                           --urlphish URL-Phish.csv --legitphish "LegitPhish Dataset.csv"
```

The script keeps only the URL and label columns. It converts PhiUSIIL to the project coding (**label 1 = phishing**; the release codes 1 = legitimate), and it repairs two double-encoded UTF-8 training URLs exactly as in the paper's copy. It also prints `content hash OK` for each file when the data match the paper's copy. If a dataset has been updated on Mendeley, the script warns, because the saved row indices assume the same rows in the same order.

## 3. Reproduce

```bash
bash scripts/run_all.sh          # CPU part: about 4-5 h on 2 cores (char-CNN training dominates)
```

| Step | Script | Output in `results/` | Paper |
|---|---|---|---|
| 0 Format audit | `step0_audit.py`, `step0_template_rule.py` | `step0_scheme_audit.csv`, `step0_format_rule_baselines.csv` | Table I |
| 1 LightGBM reference + IsHTTPS flip | `step1_features.py`, `step1_lgbm.py` | `step1_lgbm_performance.csv`, `step1_lgbm_IsHTTPS_flip.csv` | Table II/III |
| 2a char-CNN (4 input modes × seeds 42-44) | `step2_charcnn.py` (`scripts/run_charcnn_all.sh`) | `predictions/charcnn/*.csv.gz`, `models/*.pt` | – |
| 2b DistilBERT (GPU) | `notebooks/transformer_gpu.ipynb` → `import_transformer.py` | `predictions/transformer/*.csv.gz` | – |
| 3 E1 transfer | `step3_4_tables.py e1` | `step3_E1_transfer.csv`, `step3_E1_error_breakdown.csv` | Table II |
| 4 E2 counterfactual edits | `step4_lgbm_variants.py`, `step3_4_tables.py e2` | `step4_E2_counterfactual.csv` | Table III |
| 5 E3 attribution | `step5_attr_charcnn.py`, notebook cell 6, `step5_figure.py` | `step5_E3_summary.csv`, `step5_E3_signed_charCNN.csv`, `step5_attr_*_s42.csv.gz` | Fig. 1 |
| 6 E4 scheme/format-neutral training | `step3_4_tables.py e4 charCNN`, `… e4 DistilBERT` | `step6_E4_*_paired_delta.csv`, `step6_E4_*_performance.csv` | Table IV |
| Tables | `make_tables.py` | `tables.md`, `number_sources.csv` | Tables I–IV |

**GPU part.**
1. Run the notebook. Put the three prepared `data/*.csv` files and the three `splits/*.npz` files in one Drive folder, and set `DATA_DIR`.
2. Run `python src/import_transformer.py <output folder>`.
3. Copy `transformer_attr_distilbert-base-cased_s42.csv.gz` to `results/step5_attr_DistilBERT_s42.csv.gz`.
4. Rerun `scripts/run_all.sh`. Finished char-CNN runs are skipped.

## Revision analyses (internal review)
| Script | Output in `results/` | Question |
|---|---|---|
| `src/rev_a_analyses.py` | `rev_A1_baselines_metrics.csv`, `rev_A4_*`, `rev_A6_*`, `rev_A7_saturation.csv`, `rev_A8_label_budget_threshold.csv` | trivial baselines, MCC, agreement with the template rule (Cohen's κ), registered-domain overlap, TLD profile, logit saturation, small-label threshold |
| `src/rev_b_factorial.py` | `rev_B0`–`rev_B4` | 24-cell factorial format edits (scheme × www × trailing slash × path) with training support per cell; residual false positives by path |
| `src/rev_c_transfer.py` | `rev_C_transfer*.csv` | LightGBM with format features; reverse (URL-Phish → PhiUSIIL) and mixed-source training |
| `scripts/run_charcnn_long.sh`, `src/rev_d_e4.py` | `rev_D_E4_*` | E4 with a 20-epoch cap and the host-only variant; TPR@val, MCC, label-budget thresholds, FPR by path |
| `src/rev_e_gpu_checks.py` | `rev_E1_label_coding_check.csv`, `rev_E2_*` | correct-coding DistilBERT re-run vs. the sign-corrected run; DistilBERT factorial edits |
| `src/rev_f_multidataset.py`, `src/rev_f_figure.py` | `rev_F_audit_5datasets.csv`, `rev_F_format_rule_transfer.csv`, `rev_F_cell_profile.csv`, `figures/fig2_format_transfer.*`; tool reports in `results/tool_audit/` | format audit of five datasets with **urlformat-audit** and cross-dataset transfer of the format-only rule (Table I, Fig. 1 of the paper) |

## The urlformat-audit tool

`tools/urlformat-audit/` holds the source of the pip package used for the five-dataset audit:

```bash
pip install urlformat-audit          # or: pip install -e tools/urlformat-audit
urlformat-audit audit data.csv --positive phishing
urlformat-audit compare A.csv B.csv --positive 1
```

See `tools/urlformat-audit/README.md`.

### Extra datasets for the five-dataset audit (not redistributed)

`src/rev_f_multidataset.py` additionally reads, under `data/extra/` (or `CP_EXTRA`):
- `pdd/input/data_legitimate_36400.json` and `pdd/input/data_phishing_37175.json` from https://github.com/ebubekirbbr/pdd (Sahingoz et al., 2019)
- `Using-machine-learning-to-detect-malicious-URLs/data/data.csv` from https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs (label `bad` = malicious)

## Protocol in brief

- **Split.** PhiUSIIL uses a stratified group split on exact URLs (StratifiedGroupKFold, first fold, seed 42): train 150,931 / validation 37,716 / test 47,148. The indices are stored in `splits/`.
- **Target cohorts.** Target rows whose URL appears in PhiUSIIL train or validation are dropped, as is the unlabeled LegitPhish row. This leaves URL-Phish 116,126 and LegitPhish 71,767.
- **Training and thresholds.** Training uses PhiUSIIL train only. Validation is used only for early stopping and for the FPR ≤ 1% threshold. Target labels are used only for evaluation and for the "oracle" operating point, which is reported as diagnostic.
- **Metrics:**
  - F1 and FPR at p = 0.5;
  - PR-AUC on the logit;
  - TPR at FPR ≤ 1% (validation threshold and oracle, with the achieved FPR);
  - Brier score;
  - ECE with 15 bins.
- **Uncertainty.** 95% CIs come from 1,000 class-stratified bootstrap resamples with exact URLs as the unit. The resamples are fixed per set (seeds 1001/1002/1003) and shared by all models, so differences are paired. Seeds 42, 43 and 44 are used for every model family, and the across-seed SD is reported separately.

## Reproducibility notes

- **LightGBM** is deterministic. Re-running steps 0 and 1 in a clean copy of this repository reproduced the saved results byte for byte.
- **The char-CNN** is trained on CPU with fixed seeds and 2 threads. Re-training seed 42 from this repository reproduced the paper's logits exactly (max difference 0.0) with torch 2.15.0. Other PyTorch versions or thread counts can give small numerical differences.
- **DistilBERT** was fine-tuned on a Colab Tesla T4 (torch 2.11.0, transformers 5.20.2).
  - In the paper's run the notebook read PhiUSIIL with the release coding (1 = legitimate). Because binary cross-entropy satisfies L(z, 1−y) = L(−z, y), `import_transformer.py` negates those logits. This gives exactly the phishing logit of the trained model, and it checks that the inversion is exact row by row (`results/transformer_import_check.csv`).
  - The current notebook detects the coding from the data, so new runs need no correction.
  - `notebooks/transformer_attr_only_gpu.ipynb` re-runs only the attribution step for a model trained with the inverted coding.
- **Features.** The 15 lexical features use the Public Suffix List snapshot bundled with `tldextract` 5.4.0 (no network access). Other versions can change TLD-derived features for a small number of private suffixes such as `glitch.me`.
- **Not in git.** Per-row predictions (~190 MB) and trained char-CNN weights (~32 MB) are not tracked. They are attached to the GitHub release [add link] and are re-created by the scripts.

## Citation

See `CITATION.cff`. Please also cite the three datasets.

## License

Code: MIT (see `LICENSE`). The datasets keep their own licences and are not redistributed here.
