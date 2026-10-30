#!/usr/bin/env bash
# Reproduces every CPU result of the paper in order. Prerequisite: python src/prepare_data.py ... (see README).
# The transformer (DistilBERT) part needs a GPU: run notebooks/transformer_gpu.ipynb, then
#   python src/import_transformer.py <folder with the notebook outputs>
#   cp <folder>/transformer_attr_distilbert-base-cased_s42.csv.gz results/step5_attr_DistilBERT_s42.csv.gz
# before the steps marked [needs transformer] below; without it those steps run on LightGBM + char-CNN only.
set -eu
cd "$(dirname "$0")/../src"
mkdir -p ../logs
have_tf=$(ls ../predictions/transformer/*.csv.gz >/dev/null 2>&1 && echo 1 || echo 0)
echo "== Step 0: format audit";            python step0_audit.py ../data ../splits ../results > ../logs/step0.log
python step0_template_rule.py ../data ../splits ../results >> ../logs/step0.log
echo "== Step 1: LightGBM + IsHTTPS flip"; python step1_features.py > ../logs/step1.log; python step1_lgbm.py >> ../logs/step1.log
echo "== Step 2a: char-CNN (12 models, ~3 h on 2 CPU cores)"; bash ../scripts/run_charcnn_all.sh
echo "== Step 4 support: LightGBM on edited URLs"; python step4_lgbm_variants.py > ../logs/step4_lgbm.log
if [ "$have_tf" = 1 ]; then M="LightGBM charCNN DistilBERT"; else M="LightGBM charCNN"; echo "   (no transformer predictions found: tables without DistilBERT)"; fi
echo "== Step 3: E1 transfer";            python step3_4_tables.py e1 $M > ../logs/step3_e1.log
echo "== Step 4: E2 counterfactuals";     python step3_4_tables.py e2 $M > ../logs/step4_e2.log
echo "== Step 5: char-CNN attribution";   python step5_attr_charcnn.py > ../logs/step5_attr.log
if [ -f ../results/step5_attr_DistilBERT_s42.csv.gz ]; then python step5_figure.py > ../logs/step5_fig.log; else echo "   (no DistilBERT attribution file: figure skipped)"; fi
echo "== Step 6: E4 re-training deltas";  python step3_4_tables.py e4 charCNN > ../logs/step6_e4_cnn.log
[ "$have_tf" = 1 ] && python step3_4_tables.py e4 DistilBERT > ../logs/step6_e4_db.log
if [ "$have_tf" = 1 ] && [ -f ../results/step5_attr_DistilBERT_s42.csv.gz ]; then echo "== Tables I-IV"; python make_tables.py > /dev/null; echo "   -> results/tables.md"; fi
if [ -d ../data/extra ]; then echo "== Five-dataset format audit (urlformat-audit)"; python rev_f_multidataset.py > ../logs/rev_f.log && python rev_f_figure.py >> ../logs/rev_f.log; else echo "   (data/extra missing: five-dataset audit skipped)"; fi
echo done
