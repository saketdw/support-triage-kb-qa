"""Batch question answering (Part B entry point).

    python -m kbqa.batch --input questions.csv --output answers.csv

Input: CSV with ``qid``, ``question`` and (optionally) ``as_of`` columns. A
missing/empty ``as_of`` defaults to today, with a warning — a question with no
date is a question about the present. A malformed ``as_of`` fails loudly with
the row number rather than silently answering for the wrong date.

Output: ``qid, answer, doc_ids`` (doc_ids semicolon-separated; empty when the
system abstains).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from kbqa.answer import AnswerService
from kbqa.kb import KBError


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def read_questions(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        _fail(f"input not found: {p}")
    with open(p, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = {"qid", "question"} - set(fields)
        if missing:
            _fail(f"input must have 'qid' and 'question' columns; found {fields}")
        rows = list(reader)
    if not rows:
        _fail(f"input has a header but no rows: {p}")

    today = date.today()
    out, seen = [], set()
    for i, row in enumerate(rows, start=2):  # line numbers incl. header
        qid = (row.get("qid") or "").strip()
        question = (row.get("question") or "").strip()
        if not qid:
            _fail(f"row {i}: empty qid")
        if qid in seen:
            _fail(f"row {i}: duplicate qid {qid!r}")
        seen.add(qid)
        if not question:
            _fail(f"row {i} ({qid}): empty question")
        raw = (row.get("as_of") or "").strip()
        if not raw:
            print(f"warning: row {i} ({qid}) has no as_of; assuming today ({today})", file=sys.stderr)
            as_of = today
        else:
            try:
                as_of = date.fromisoformat(raw)
            except ValueError:
                _fail(f"row {i} ({qid}): as_of is not an ISO date: {raw!r}")
        out.append({"qid": qid, "question": question, "as_of": as_of})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV with qid, question, as_of columns")
    ap.add_argument("--output", required=True, help="where to write answers.csv")
    ap.add_argument("--kb", default=None, help="knowledge-base directory (default: repo kb/)")
    args = ap.parse_args(argv)

    questions = read_questions(args.input)
    try:
        service = AnswerService(args.kb)
    except KBError as e:
        # A malformed or missing knowledge base is an operator error, not a
        # crash: name the problem instead of printing a traceback.
        _fail(f"knowledge base could not be loaded: {e}")

    # Counter, not a fixed dict: adding an answer path must never break the CLI.
    kinds: Counter[str] = Counter()
    try:
        out = open(args.output, "w", newline="", encoding="utf-8")
    except OSError as e:
        _fail(f"cannot write to --output {args.output}: {e.strerror}")
    with out as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer", "doc_ids"])
        for q in questions:
            result = service.answer(q["question"], q["as_of"])
            kinds[result.kind] += 1
            writer.writerow([q["qid"], result.answer, ";".join(result.doc_ids)])

    breakdown = ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items()))
    print(f"wrote {len(questions)} answers to {args.output} ({breakdown})")


if __name__ == "__main__":
    main()
