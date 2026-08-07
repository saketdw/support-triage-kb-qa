"""Route-classification CLI (Part A entry point).

    python -m triage.predict --input messages.csv --output predictions.csv

Reads a CSV with a ``text`` column and writes exactly ``text,prediction``
(the schema the exercise specifies). The model is trained once at startup on
``data/train.csv`` (~1s) — deterministic, no persisted artifacts to go stale.

Optional selective prediction: ``--review-threshold 0.7`` adds a
``needs_review`` column flagging rows whose top-class probability falls below
the threshold, for a human-review lane. Off by default so the default output
schema matches the brief byte-for-byte.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from triage.model import DEFAULT_TRAIN, build_pipeline, load_training


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def read_messages(path: str | Path) -> list[str]:
    """Read and validate the input CSV; returns the text column in order."""
    p = Path(path)
    if not p.exists():
        _fail(f"input not found: {p}")
    with open(p, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "text" not in reader.fieldnames:
            _fail(f"input must have a 'text' column; found columns: {reader.fieldnames}")
        rows = [(row.get("text") or "").strip() for row in reader]
    if not rows:
        _fail(f"input has a header but no rows: {p}")
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV with a 'text' column")
    ap.add_argument("--output", required=True, help="where to write predictions.csv")
    ap.add_argument("--train", default=str(DEFAULT_TRAIN), help="labeled training CSV (default: data/train.csv)")
    ap.add_argument(
        "--review-threshold", type=float, default=None, metavar="P",
        help="optional: add a needs_review column flagging predictions with confidence < P",
    )
    args = ap.parse_args(argv)

    try:
        texts, labels = load_training(args.train)
    except (FileNotFoundError, ValueError) as e:
        _fail(str(e))
    messages = read_messages(args.input)

    pipe = build_pipeline().fit(texts, labels)
    predictions = pipe.predict(messages)

    header = ["text", "prediction"]
    flags = None
    if args.review_threshold is not None:
        confidence = pipe.predict_proba(messages).max(axis=1)
        flags = confidence < args.review_threshold
        header.append("needs_review")

    try:
        out = open(args.output, "w", newline="", encoding="utf-8")
    except OSError as e:
        _fail(f"cannot write to --output {args.output}: {e.strerror}")
    with out as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (message, prediction) in enumerate(zip(messages, predictions)):
            row = [message, prediction]
            if flags is not None:
                row.append(str(bool(flags[i])).lower())
            writer.writerow(row)
    print(f"wrote {len(messages)} predictions to {args.output}")


if __name__ == "__main__":
    main()
