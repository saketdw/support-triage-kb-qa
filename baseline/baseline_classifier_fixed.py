"""Support-ticket route classifier — the reviewed fix.

A minimal diff on ``baseline_classifier.py``. What changed and why:

1. The vectorizer now lives INSIDE a Pipeline, so it is fitted only on
   training folds. The original called fit_transform on all 400 rows and
   split afterwards — the evaluation saw the test set (measured in the EDA
   notebook, section 3).
2. Evaluation is template-grouped 5-fold cross-validation with per-class
   metrics. The data is template-generated (28 near-duplicate rows); a
   random split scores the model on phrasings it memorized, and accuracy
   alone hides the class that matters — fraud-report is 12.5% of rows and
   the costliest error.
3. class_weight='balanced' aligns the loss with the stated cost asymmetry
   (a fraud error weighs ~3.2x a general one). min_df=2 drops the 387 terms
   that appear in a single document — memorization hooks (measured harmless,
   EDA section 11).
4. predict() fits once and serves many. The original retrained the full
   model on every call — and deployed a different model than it evaluated.

    python3 baseline/baseline_classifier_fixed.py
"""
import csv
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
ROUTES = ["account-access", "fraud-report", "general", "transaction-dispute"]

# Messages are template-generated with only the asset name swapped. Grouping by
# the asset-masked, normalized text keeps every phrasing of one template on the
# same side of a CV split, so the model is always scored on unseen phrasings.
_ASSET = re.compile(
    r"\b(btc|eth|sol|ada|doge|dogecoin|bitcoin|ethereum|solana|polygon|matic|"
    r"xrp|usdc|ltc|litecoin|avax|cardano|tron|dot|polkadot)\b"
)


def load(path=DATA):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def template_groups(texts):
    """Group id per message: normalized text with asset names masked."""
    seen = {}
    out = []
    for t in texts:
        key = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", t.lower())).strip()
        out.append(seen.setdefault(_ASSET.sub("ASSET", key), len(seen)))
    return np.array(out)


def build_pipeline():
    """Vectorizer INSIDE the pipeline: fitting on test folds is impossible."""
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced"),
    )


def main():
    texts, labels = load()
    print(f"loaded {len(texts)} rows")

    preds = cross_val_predict(
        build_pipeline(),
        np.array(texts, dtype=object),
        np.array(labels),
        cv=GroupKFold(n_splits=5),
        groups=template_groups(texts),
    )
    print("template-grouped 5-fold cross-validation:\n")
    print(classification_report(labels, preds, digits=3))
    print(f"confusion matrix (rows=true, cols=pred): {ROUTES}")
    print(confusion_matrix(labels, preds, labels=ROUTES))
    fraud = recall_score(labels, preds, labels=["fraud-report"], average=None)[0]
    print(f"\nfraud-report recall (the metric production should gate on): {fraud:.3f}")
    return fraud


_FITTED = None


def predict(text):
    """predict(text) -> route label. Fits once, serves many."""
    global _FITTED
    if _FITTED is None:
        texts, labels = load()
        _FITTED = build_pipeline().fit(texts, labels)
    return _FITTED.predict([text])[0]


if __name__ == "__main__":
    main()
