# CODSOFT_TASK4 — Spam SMS Detection

A machine-learning project that classifies SMS messages as **spam** or **ham**
(legitimate), using TF-IDF text features and classic classifiers. Built for the
**CodSoft Machine Learning Internship (Task 4)**.

## Overview

Each message is cleaned (lower-cased, links collapsed to `url`, punctuation
removed) and turned into **TF-IDF** features (unigrams + bigrams). Three
classifiers are compared:

| Model | Notes |
|---|---|
| Multinomial Naive Bayes | classic spam baseline |
| Logistic Regression | linear, gives probabilities |
| Linear SVM | calibrated to output probabilities |

The best model (by spam F1) is saved and reused to classify new messages.

## Project structure

```
CODSOFT_TASK4/
├── train.py           # train + evaluate + save the best model
├── predict.py         # classify new messages
├── requirements.txt
├── sample_data.csv    # small synthetic demo dataset
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --data sample_data.csv
python train.py --data SMSSpamCollection        # UCI tab-separated file
python train.py --data spam.csv                 # Kaggle spam.csv (v1,v2 columns)
```

Artefacts land in `models/`:

* `best_model.joblib` — vectoriser + classifier
* `metrics.json` — accuracy / precision / recall / F1 / ROC-AUC per model
* `model_comparison.png`, `confusion_matrix.png`, `roc_curves.png`

## Predict

```bash
python predict.py --text "Congratulations! You've won a free ticket, call now!"
python predict.py --file messages.txt
```

## Results

On the **SMS Spam Collection** (5,574 messages, 13.4% spam), 80/20 stratified
split:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Multinomial Naive Bayes | 0.968 | 1.000 | 0.758 | 0.863 | 0.985 |
| Logistic Regression | 0.982 | 0.985 | 0.879 | 0.929 | 0.989 |
| **Linear SVM** (best) | **0.984** | 0.928 | **0.953** | **0.940** | **0.990** |

All three models separate spam very well (ROC-AUC ≥ 0.985). The SVM gives the
best balance of precision and recall. For a spam filter you often care most
about **recall** (don't miss spam) while keeping precision high — tweak the
decision threshold in `predict.py` to trade one for the other.

## Dataset

* **UCI / Kaggle:** "SMS Spam Collection" — the tab-separated
  `SMSSpamCollection` file, or Kaggle's `spam.csv` (columns `v1`, `v2`).
* Any CSV with a label column (`spam`/`ham`) and a message column also works.

## Notes

* The loader auto-detects the tab-separated UCI format and the Kaggle
  `v1`/`v2` CSV format.
* URL patterns are collapsed to a single `url` token so the model learns
  "contains a link" rather than memorising specific domains.
