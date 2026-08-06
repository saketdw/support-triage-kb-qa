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

_ASSET = re.compile(
    r"\b(btc|eth|sol|ada|doge|dogecoin|bitcoin|ethereum|solana|polygon|matic|"
    r"xrp|usdc|ltc|litecoin|avax|cardano|tron|dot|polkadot)\b"
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
    """Group id per message: normalized text with asset names masked.

    Used by evaluation code (GroupKFold) so sibling phrasings of one template
    never straddle a train/test boundary.
    """
    seen: dict[str, int] = {}
    out = []
    for t in texts:
        key = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", t.lower())).strip()
        out.append(seen.setdefault(_ASSET.sub("ASSET", key), len(seen)))
    return np.array(out)
