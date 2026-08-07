"""Part C smoke tests. Skipped when tesseract isn't installed (optional
extra) — e.g. on CI; run locally after `brew install tesseract`."""
import csv
import shutil
from pathlib import Path

import pytest

pytesseract = pytest.importorskip("pytesseract")
pytestmark = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not installed (optional extra)"
)

from partc import ocr_route  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_all_three_screenshots_route_correctly(tmp_path):
    out = tmp_path / "routed.csv"
    ocr_route.main(["--input", str(ROOT / "media" / "screenshots"), "--output", str(out)])
    rows = {r["file"]: r for r in csv.DictReader(open(out, newline="", encoding="utf-8"))}
    assert rows["login-error.png"]["route"] == "account-access"
    assert rows["phishing-sms.png"]["route"] == "fraud-report"
    assert rows["txn-failed.png"]["route"] == "transaction-dispute"
    # clean synthetic renders should OCR confidently
    assert all(float(r["ocr_confidence"]) > 80 for r in rows.values())


def test_covariate_shift_is_caught_by_the_review_band(tmp_path):
    """The phishing SMS routes correctly but with low classifier confidence —
    OCR text (UI chrome + SMS thread) is off the training distribution. The
    review band exists precisely for this ticket."""
    out = tmp_path / "routed.csv"
    ocr_route.main(["--input", str(ROOT / "media" / "screenshots"), "--output", str(out)])
    rows = {r["file"]: r for r in csv.DictReader(open(out, newline="", encoding="utf-8"))}
    assert rows["phishing-sms.png"]["needs_review"] == "true"
    assert float(rows["phishing-sms.png"]["route_confidence"]) < 0.5
