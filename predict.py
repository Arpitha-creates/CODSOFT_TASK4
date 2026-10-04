#!/usr/bin/env python3
"""
CODSOFT Machine Learning Internship -- Task 4
Classify SMS messages as spam or ham.

Usage
-----
    python predict.py --text "Congratulations! You've won a free ticket, call now!"
    python predict.py --file messages.txt          # one message per line
"""

from __future__ import annotations

import argparse

import joblib

from train import clean_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Spam SMS predictor")
    ap.add_argument("--text", help="a single message")
    ap.add_argument("--file", help="a text file with one message per line")
    ap.add_argument("--model", default="models/best_model.joblib")
    args = ap.parse_args()

    if not args.text and not args.file:
        ap.error("provide either --text or --file")

    bundle = joblib.load(args.model)
    model = bundle["pipeline"]
    print(f"[info] using model: {bundle.get('model_name', 'unknown')}\n")

    def show(msg: str) -> None:
        proba = model.predict_proba([clean_text(msg)])[0][1]
        label = "SPAM" if proba >= 0.5 else "ham"
        print(f"[{label:>4}] (spam prob {proba:.2f})  {msg[:90]}")

    if args.text:
        show(args.text)
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    show(line)


if __name__ == "__main__":
    main()
