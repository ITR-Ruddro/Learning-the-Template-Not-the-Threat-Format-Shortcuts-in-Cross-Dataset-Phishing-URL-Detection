# urlformat-audit

Check a labelled phishing / malicious URL dataset for **format shortcuts**: URL formatting
(scheme, `www.`, trailing slash, path) that differs between the classes and lets a model
predict the label without learning anything about the threat.

Example: in some datasets every legitimate URL is written `https://www.<host>` and phishing
URLs are not. A model trained on such data learns "legitimate = this template". It scores
well in-dataset and fails on any other dataset.

`urlformat-audit` reports:

| Output | Meaning |
|---|---|
| **Class table** | % of URLs per class with each format property: `https`, `http`, no scheme, `www.`, trailing slash, path, query, port, IP host, uppercase, median length |
| **Format cells** | the 24 cells scheme (https/http/none) × `www.` × trailing slash × path, with class counts and purity |
| **Format-only rule** | predicts the majority label of each cell. Its MCC says how much of the label format alone explains. MCC ≥ 0.5 is flagged. |
| **Template rule** | "legitimate iff the URL matches `^https://www\.[a-z0-9.-]+$`" |
| **Trivial baseline** | predict the positive class everywhere |
| **NMI** | I(cell; label) / H(label): share of label entropy explained by the format cell (0 = none, 1 = all) |
| **Flags** | properties whose prevalence differs by ≥ 30 percentage points between the classes, and large single-class cells |
| **compare** | fit the format-only rule on dataset A, evaluate it on dataset B. This shows whether format cues transfer, fail, or invert. |
| **factorial** | writes the 24 format variants of every URL, for testing a trained model's sensitivity to formatting |

Requires Python ≥ 3.9 with numpy and pandas. Nothing else.

## Install

```bash
pip install urlformat-audit
```

## Command line

```bash
# one file with a label column (the positive class is the phishing/malicious label)
urlformat-audit audit data.csv --positive phishing
urlformat-audit audit data.csv --url-col URL --label-col label --positive 1 --json report.json

# one file per class, without a label column (CSV/TSV with a url column, or a JSON list of URLs)
urlformat-audit audit legit.json phish.json --label 0 1 --name "my dataset"

# does a format-only rule fitted on A transfer to B?
urlformat-audit compare A.csv B.csv --positive 1

# 24 format variants of every URL (columns: id = row of the input, cell, url_variant)
urlformat-audit factorial test.csv -o test_factorial.csv
```

Example output (dataset of Sahingoz et al., 2019; legitimate = 0, phishing = 1), abridged:

```
Format-only rule (majority per cell):   accuracy=0.756  balanced_accuracy=0.756  f1_positive=0.771  mcc=0.515  tpr=0.812  fpr=0.300
Template rule (legit iff ^https://www.<host>$): accuracy=0.505  ...  mcc=0.000
Normalized mutual information I(cell;label)/H(label) = 0.266

Shortcut flags:
  - scheme=https: 36.8% of negatives vs 1.8% of positives
  - scheme=http: 63.2% of negatives vs 97.7% of positives
  - format-only rule (4 cues, 24 cells) reaches MCC 0.515 / balanced accuracy 0.756 on this dataset
```

## Python API

```python
from urlformat_audit import audit, compare, factorial_variants, format_features, FormatRule

r = audit(urls, labels, name="my data")   # labels: 1 = phishing/malicious, 0 = legitimate
print(r.format_rule["mcc"], r.nmi_cell_label)
print(r.class_table)       # pandas DataFrame
print(r.cell_table.head())
for f in r.flags: print(f)
r.to_dict()                # JSON-serialisable

res = compare(train_urls, train_labels, test_urls, test_labels)
res["format_rule_transfer"]["mcc"], res["share_target_in_unseen_cells"]

factorial_variants("http://example.org/login/")   # [(cell_name, url), ...] x 24
format_features(urls)                              # per-URL format columns + cell
```

## How to read the result

- **Format-only rule MCC near 1** (or a NMI near 1): format alone separates the classes, so any
  model that sees the raw URL can learn the shortcut. Cross-dataset results on this data need
  format-controlled checks: factorial edits, host-only input, or a scheme-neutral representation.
- **High in-dataset MCC but low or negative `compare` MCC**: the format cue is specific to that dataset.
  Expect transfer failures.
- **Low MCC**: format is not a major shortcut. This says nothing about other artifacts (domains, TLDs,
  collection time).

The audit is descriptive. It cannot tell you whether a format difference is a collection
artifact or a real property of phishing URLs; it only tells you that a model can use it.

## Development

```bash
pip install -e ".[test]"
pytest
python -m build        # sdist + wheel in dist/
```

## Citation

If you use this tool, please cite the accompanying paper:
*Learning the Template, Not the Threat: Format Shortcuts in Cross-Dataset Phishing URL Detection* (authors and venue to be added).

## License

MIT
