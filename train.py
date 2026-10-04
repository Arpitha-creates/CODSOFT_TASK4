#!/usr/bin/env python3
"""
CODSOFT Machine Learning Internship
Task 4 -- Spam SMS Detection
=============================

Classifies SMS messages as **spam** or **ham** (legitimate) using TF-IDF
features and classic classifiers.

Models compared:
    * Multinomial Naive Bayes
    * Logistic Regression
    * Linear SVM (probability-calibrated)

Usage
-----
    python train.py --data sample_data.csv
    python train.py --data spam.csv
    python train.py --data SMSSpamCollection      # UCI tab-separated file

Outputs (into --outdir, default ./models)
-----------------------------------------
    best_model.joblib       best pipeline (vectoriser + classifier)
    metrics.json            accuracy / precision / recall / F1 / ROC-AUC per model
    model_comparison.png    bar chart comparing the models
    confusion_matrix.png    confusion matrix of the best model
    roc_curves.png          ROC curves of all models
"""

from __future__ import annotations

import argparse
import json
import os
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)     # collapse links
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: str, label_col: str | None = None,
                 text_col: str | None = None):
    # Detect the UCI tab-separated format (label<TAB>message, no header).
    with open(path, encoding="utf-8", errors="ignore") as fh:
        first = fh.readline()
    if "\t" in first:                       # UCI tab-separated file
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"],
                         quoting=3, encoding="utf-8", engine="python")
        # Some copies carry a header row ("label\tmessage") - drop it if present.
        if str(df.iloc[0, 0]).strip().lower() in {"label", "v1", "type", "category"}:
            df = df.iloc[1:].reset_index(drop=True)
    else:
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        low = {c.lower(): c for c in df.columns}
        if label_col is None and "v1" in low:          # Kaggle spam.csv
            label_col, text_col = low["v1"], low["v2"]
        elif label_col is None and "label" in low:
            label_col, text_col = low["label"], low.get("message", low.get("v2"))

    df = df.dropna().reset_index(drop=True)
    if label_col is None or text_col is None:
        str_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
        label_col = min(str_cols, key=lambda c: df[c].nunique())
        text_col = max([c for c in str_cols if c != label_col],
                       key=lambda c: df[c].astype(str).str.len().mean())

    print(f"[data] label='{label_col}', text='{text_col}'")
    y = df[label_col].astype(str).str.strip().str.lower().map(
        lambda v: 1 if v in {"spam", "1", "yes", "true"} else 0)
    X = df[text_col].map(clean_text)
    print(f"[data] {len(X)} messages, spam rate {y.mean():.1%}")
    return X, y


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_models(max_features: int) -> dict[str, Pipeline]:
    def tfidf():
        return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                               sublinear_tf=True, stop_words="english")

    return {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", tfidf()), ("clf", MultinomialNB())]),
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=1000, C=5,
                                       random_state=RANDOM_STATE))]),
        "Linear SVM": Pipeline([
            ("tfidf", tfidf()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(random_state=RANDOM_STATE), cv=5))]),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_comparison(results: dict, outpath: str) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    names = list(results)
    x = np.arange(len(metrics))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, n in enumerate(names):
        ax.bar(x + i * w - 0.4 + w / 2, [results[n][m] for m in metrics], w, label=n)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Spam SMS Detection - model comparison")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


def plot_confusion(cm, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ham", "Spam"]); ax.set_yticklabels(["Ham", "Spam"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix - best model")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


def plot_roc(curves: dict, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Spam SMS classifier trainer")
    ap.add_argument("--data", default="sample_data.csv")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-features", type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X, y = load_dataset(args.data, args.label_col, args.text_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y)

    models = build_models(args.max_features)
    results, curves, fitted = {}, {}, {}
    for name, pipe in models.items():
        print(f"[train] {name} ...")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
        }
        fpr, tpr, _ = roc_curve(y_test, proba)
        curves[name] = (fpr, tpr, results[name]["roc_auc"])
        fitted[name] = pipe
        print("        " + "  ".join(f"{k}={v}" for k, v in results[name].items()))

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted[best_name]
    print(f"\n[best] {best_name} (f1={results[best_name]['f1']})")

    preds = best_model.predict(X_test)
    report = classification_report(y_test, preds, target_names=["ham", "spam"],
                                   zero_division=0)
    print("\n" + report)

    joblib.dump({"pipeline": best_model, "model_name": best_name,
                 "classes": ["ham", "spam"]},
                os.path.join(args.outdir, "best_model.joblib"))
    with open(os.path.join(args.outdir, "metrics.json"), "w") as fh:
        json.dump({"results": results, "best_model": best_name,
                   "classification_report": report}, fh, indent=2)

    plot_comparison(results, os.path.join(args.outdir, "model_comparison.png"))
    plot_confusion(confusion_matrix(y_test, preds),
                   os.path.join(args.outdir, "confusion_matrix.png"))
    plot_roc(curves, os.path.join(args.outdir, "roc_curves.png"))
    print(f"\n[done] artefacts written to '{args.outdir}/'")


if __name__ == "__main__":
    main()
