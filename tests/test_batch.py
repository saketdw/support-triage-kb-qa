"""Batch CLI: schema round-trip and input validation."""
import csv
from pathlib import Path

import pytest

from kbqa import batch

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows, header=("qid", "question", "as_of")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_roundtrip_schema_and_order(tmp_path):
    inp, out = tmp_path / "q.csv", tmp_path / "a.csv"
    write_csv(inp, [
        ["x1", "What fee do I pay to withdraw crypto from my account?", "2026-07-28"],
        ["x2", "What was the crypto withdrawal fee in June 2025?", "2025-06-01"],
        ["x3", "Can I pay my electricity bill through the app?", "2026-07-28"],
    ])
    batch.main(["--input", str(inp), "--output", str(out)])
    rows = list(csv.reader(open(out, newline="", encoding="utf-8")))
    assert rows[0] == ["qid", "answer", "doc_ids"]
    assert [r[0] for r in rows[1:]] == ["x1", "x2", "x3"]  # order preserved
    assert rows[1][2] == "kb-013" and "0.4%" in rows[1][1]
    assert rows[2][2] == "kb-012" and "0.9%" in rows[2][1]
    assert all(len(r) == 3 for r in rows[1:])


def test_missing_as_of_defaults_to_today_with_warning(tmp_path, capsys):
    inp, out = tmp_path / "q.csv", tmp_path / "a.csv"
    write_csv(inp, [["x1", "How long does a bank deposit take to clear?"]], header=("qid", "question"))
    batch.main(["--input", str(inp), "--output", str(out)])
    assert "assuming today" in capsys.readouterr().err
    rows = list(csv.reader(open(out, newline="", encoding="utf-8")))
    assert len(rows) == 2


@pytest.mark.parametrize("rows,header,match", [
    ([["x1", "q", "2026-13-45"]], ("qid", "question", "as_of"), "not an ISO date"),
    ([["x1", "", "2026-07-28"]], ("qid", "question", "as_of"), "empty question"),
    ([["", "q", "2026-07-28"]], ("qid", "question", "as_of"), "empty qid"),
    ([["x1", "q", "2026-07-28"], ["x1", "q2", "2026-07-28"]], ("qid", "question", "as_of"), "duplicate qid"),
])
def test_bad_rows_fail_loudly(tmp_path, capsys, rows, header, match):
    inp = tmp_path / "q.csv"
    write_csv(inp, rows, header=header)
    with pytest.raises(SystemExit) as e:
        batch.main(["--input", str(inp), "--output", str(tmp_path / "a.csv")])
    assert e.value.code == 2
    assert match in capsys.readouterr().err


def test_missing_question_column_fails_loudly(tmp_path, capsys):
    inp = tmp_path / "q.csv"
    inp.write_text("id,text\n1,hello\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        batch.main(["--input", str(inp), "--output", str(tmp_path / "a.csv")])
    assert e.value.code == 2


def test_missing_input_file_fails_loudly(tmp_path):
    with pytest.raises(SystemExit) as e:
        batch.main(["--input", str(tmp_path / "nope.csv"), "--output", str(tmp_path / "a.csv")])
    assert e.value.code == 2
