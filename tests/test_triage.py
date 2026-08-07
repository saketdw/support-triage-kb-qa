"""Part A tests: the fixed classifier and the predict CLI."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from triage import model, predict

ROOT = Path(__file__).resolve().parents[1]


def _load_fixed_baseline():
    spec = importlib.util.spec_from_file_location(
        "baseline_fixed", ROOT / "baseline" / "baseline_classifier_fixed.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_is_leakage_proof():
    """The vectorizer must live inside the Pipeline — the structural fix for
    the original bug (vectorizer fitted on the full corpus before the split)."""
    pipe = model.build_pipeline()
    assert isinstance(pipe, Pipeline)
    steps = [s for _, s in pipe.steps]
    assert isinstance(steps[0], TfidfVectorizer)
    assert isinstance(steps[-1], LogisticRegression)


def test_cli_config_matches_reviewed_fix():
    """triage.model must stay identical to the reviewed small-diff fix in
    baseline/baseline_classifier_fixed.py — no silent drift."""
    fixed = _load_fixed_baseline()
    ours, theirs = model.build_pipeline(), fixed.build_pipeline()
    for key in ["ngram_range", "min_df", "sublinear_tf"]:
        assert ours.steps[0][1].get_params()[key] == theirs.steps[0][1].get_params()[key]
    for key in ["C", "class_weight", "max_iter"]:
        assert ours.steps[-1][1].get_params()[key] == theirs.steps[-1][1].get_params()[key]


def test_template_grouping_recovers_the_generator_families():
    """Grouping must collapse all three wrapper slots (asset, greeting,
    closing), not just the asset — otherwise phrasings of one body land in
    different groups and still straddle a split."""
    texts, _ = model.load_training()
    groups = model.template_groups(texts)
    n_groups, biggest = len(set(groups)), np.bincount(groups).max()
    assert n_groups < 200, f"grouping too weak: {n_groups} groups (expected ~169 families)"
    assert biggest >= 8, f"largest family only {biggest} rows; wrapper slots are not collapsed"

    # a body template written with different wrappers must share one group
    probes = [
        "Hi, Can you explain how to move Ethereum to an external wallet?",
        "Quick question, Can you explain how to move XRP to an external wallet? Thanks.",
        "Can you explain how to move BTC to an external wallet? Please advise.",
    ]
    ids = model.template_groups(probes)
    assert len(set(ids)) == 1, f"wrapper variants split across groups: {ids}"


def test_grouped_cv_fraud_recall_floor():
    """Fraud-report recall on an entirely unseen phrasing family. Measured
    0.78; the floor guards against regression, and the honest headline is the
    range 0.76-1.00 (see the notebook, section 5) — not this number alone."""
    texts, labels = model.load_training()
    preds = cross_val_predict(
        model.build_pipeline(),
        np.array(texts, dtype=object),
        np.array(labels),
        cv=GroupKFold(n_splits=5),
        groups=model.template_groups(texts),
    )
    fraud = recall_score(labels, preds, labels=["fraud-report"], average=None)[0]
    assert fraud >= 0.75, f"fraud-report recall regressed: {fraud:.3f}"


def test_review_band_separates_errors_from_correct_predictions():
    """Selective prediction is the design that makes 0.78 shippable: a
    low-confidence band must catch the errors it is meant to catch."""
    texts, labels = model.load_training()
    labels_arr = np.array(labels)
    proba = cross_val_predict(
        model.build_pipeline(),
        np.array(texts, dtype=object),
        labels_arr,
        cv=GroupKFold(n_splits=5),
        groups=model.template_groups(texts),
        method="predict_proba",
    )
    conf = proba.max(axis=1)
    wrong = np.array(sorted(set(labels)))[proba.argmax(axis=1)] != labels_arr
    assert np.median(conf[wrong]) < np.median(conf[~wrong]), "confidence does not separate errors"
    band = conf < 0.70
    caught = (band & wrong).sum() / wrong.sum()
    assert caught >= 0.9, f"review band catches only {caught:.0%} of errors"
    assert band.mean() <= 0.35, f"review band sends {band.mean():.0%} of traffic to humans"


@pytest.fixture()
def messages_csv(tmp_path):
    p = tmp_path / "messages.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text"])
        w.writerow(["I can't log into my account, the password reset never arrives."])
        w.writerow(["Someone moved money out of my account without my permission."])
        w.writerow(['Hello, my order shows "completed, but refunded" - what does that mean?'])
        w.writerow(["How does staking work on Ethereum?"])
    return p


def test_cli_roundtrip_schema(messages_csv, tmp_path):
    out = tmp_path / "predictions.csv"
    predict.main(["--input", str(messages_csv), "--output", str(out)])
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["text", "prediction"], "output schema must be exactly text,prediction"
    assert len(rows) == 5  # header + 4 messages, order preserved
    assert rows[1][0].startswith("I can't log into")
    valid = {"account-access", "fraud-report", "general", "transaction-dispute"}
    assert all(r[1] in valid for r in rows[1:])


def test_cli_deterministic(messages_csv, tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    predict.main(["--input", str(messages_csv), "--output", str(a)])
    predict.main(["--input", str(messages_csv), "--output", str(b)])
    assert a.read_bytes() == b.read_bytes()


def test_cli_review_flag(messages_csv, tmp_path):
    out = tmp_path / "flagged.csv"
    predict.main(["--input", str(messages_csv), "--output", str(out), "--review-threshold", "0.99"])
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["text", "prediction", "needs_review"]
    assert all(r[2] in {"true", "false"} for r in rows[1:])


def test_cli_rejects_missing_text_column(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("message\nhello\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        predict.main(["--input", str(bad), "--output", str(tmp_path / "out.csv")])
    assert e.value.code == 2


def test_cli_rejects_missing_file(tmp_path):
    with pytest.raises(SystemExit) as e:
        predict.main(["--input", str(tmp_path / "nope.csv"), "--output", str(tmp_path / "out.csv")])
    assert e.value.code == 2


def test_cli_unwritable_output_fails_cleanly(messages_csv, tmp_path):
    """Operator errors must exit with a message, not a traceback."""
    with pytest.raises(SystemExit) as e:
        predict.main(["--input", str(messages_csv), "--output", "/nonexistent-dir/out.csv"])
    assert e.value.code == 2
