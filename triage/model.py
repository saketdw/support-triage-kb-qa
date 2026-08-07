"""Canonical Part A model configuration and data helpers.

This mirrors ``baseline/baseline_classifier_fixed.py`` — the reviewed
small-diff fix — and a test asserts the two configurations stay identical,
so the CLI can never silently drift from the reviewed model.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = REPO_ROOT / "data" / "train.csv"

# The corpus is generated from ~169 body templates wrapped in four slots:
#   {greeting} + {body} + {asset} + {closing}
# Collapsing all three wrapper slots recovers the real template family. Masking
# only the asset (the obvious first attempt) finds 386 "groups" of at most 2 —
# it leaves greeting/closing variants of one body in *different* groups, so they
# still straddle a split. See notebooks/01_eda_baseline_review.ipynb section 2.
_ASSET = re.compile(
    r"\b(btc|eth|sol|ada|doge|dogecoin|bitcoin|ethereum|solana|polygon|matic|"
    r"xrp|usdc|ltc|litecoin|avax|cardano|tron|dot|polkadot)\b"
)
_GREETING = re.compile(
    r"^(hi|hey|hello team|hello|good morning|quick question|please help|dear support)\b[ ,]*"
)
_CLOSING = re.compile(
    r"\b(thanks|thank you|please advise|appreciate any help|this is time sensitive|"
    r"any help appreciated)\b[ .]*"
)


def load_training(path: str | Path = DEFAULT_TRAIN) -> tuple[list[str], list[str]]:
    """Load and validate the labeled training CSV (columns: text, label)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"training data not found: {p}")
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or {"text", "label"} - set(reader.fieldnames):
            raise ValueError(
                f"training CSV must have 'text' and 'label' columns; found {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"training CSV is empty: {p}")
    return [r["text"] for r in rows], [r["label"] for r in rows]


def build_pipeline() -> Pipeline:
    """The reviewed configuration. Vectorizer inside the pipeline: fitting
    preprocessing on evaluation folds is structurally impossible."""
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced"),
    )


def template_groups(texts: list[str]) -> np.ndarray:
    """Group id per message, keyed on the generator's *body* template.

    Normalizes the text, then strips the three variable wrapper slots (asset,
    greeting, closing) so that every phrasing of one body shares an id. Used by
    evaluation code with GroupKFold, which then scores the model only on
    template families it never trained on.

    Note what this measures: grouped CV answers "how do we do on an entirely
    new phrasing family?", while a stratified split answers "how do we do if
    next week resembles this corpus?". Both are reported — the gap between them
    is the memorization component (see the notebook, sections 5 and 8).
    """
    seen: dict[str, int] = {}
    out = []
    for t in texts:
        key = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", t.lower())).strip()
        key = _ASSET.sub("ASSET", key)
        key = _CLOSING.sub("", _GREETING.sub("", key))
        out.append(seen.setdefault(re.sub(r"\s+", " ", key).strip(), len(seen)))
    return np.array(out)
