"""Golden regression on the real KB — the canary suite.

These pin the exact behaviors the exercise cares about, on the shipped
knowledge base. In production this same file re-runs on every KB change and
diffs the cited doc_ids (eval/gold.csv is the full version).
"""
import csv
import re
from datetime import date
from pathlib import Path

import pytest

from kbqa.answer import AnswerService

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def svc():
    return AnswerService(ROOT / "kb")


@pytest.fixture(scope="module")
def questions():
    return list(csv.DictReader(open(ROOT / "questions.csv", encoding="utf-8")))


def test_invariant_no_answer_cites_an_out_of_window_doc(svc, questions):
    """Across ALL provided questions: every cited document was in force at the
    question's as_of — except negatives, which cite the expired doc as
    evidence. This is what makes wrong-version answers structurally
    impossible."""
    for q in questions:
        as_of = date.fromisoformat(q["as_of"])
        a = svc.answer(q["question"], as_of)
        for doc_id in a.doc_ids:
            doc = svc.kb.docs[doc_id]
            if a.kind == "negative":
                assert not doc.in_force(as_of)
            else:
                assert doc.in_force(as_of), (
                    f"{q['qid']}: cited {doc_id} which was not in force on {as_of}"
                )


def test_current_fee_question_gets_the_current_doc_and_fact(svc):
    a = svc.answer("What fee do I pay to withdraw crypto from my account?", date(2026, 7, 28))
    assert a.doc_ids[0] == "kb-013"
    assert "0.4%" in a.answer
    assert "1.5%" not in a.answer and "0.9%" not in a.answer  # stale figures


def test_no_minimum_charge_today(svc):
    a = svc.answer("What is the minimum charge on a withdrawal?", date(2026, 7, 28))
    assert a.doc_ids[0] == "kb-013"
    assert "no minimum" in a.answer.lower()


def test_historical_fee_question_gets_the_superseded_doc(svc):
    a = svc.answer("What was the crypto withdrawal fee in June 2025?", date(2025, 6, 1))
    assert a.doc_ids[0] == "kb-012"
    assert "0.9%" in a.answer
    assert "0.4%" not in a.answer  # today's figure must not leak into 2025


def test_historical_2fa_question_gets_the_sms_era_procedure(svc):
    a = svc.answer(
        "If I lost my authenticator in September 2025, how was I supposed to recover access?",
        date(2025, 9, 1),
    )
    assert a.doc_ids[0] == "kb-021"
    assert "SMS" in a.answer


def test_expired_maintenance_notice_is_a_negative(svc):
    a = svc.answer("Are withdrawals currently paused for maintenance?", date(2026, 7, 28))
    assert a.kind == "negative"
    assert a.doc_ids == ("kb-091",)
    assert "2026-06-14" in a.answer


def test_ended_promotion_is_a_negative(svc):
    a = svc.answer("Is the invite a friend promotion still running?", date(2026, 7, 28))
    assert a.kind == "negative"
    assert a.doc_ids == ("kb-092",)


def test_phishing_question_cites_the_phishing_doc(svc):
    a = svc.answer("Support emailed me asking for my one time code, is that legitimate?", date(2026, 7, 28))
    assert "kb-110" in a.doc_ids


def test_fraud_phone_number_is_never_invented(svc):
    """The hallucination trap: retrieval correctly finds the fraud doc, but no
    phone number exists anywhere in the KB. The answer must name the gap and
    must not contain anything phone-number-shaped."""
    a = svc.answer("What is the direct phone number for your fraud team?", date(2026, 7, 28))
    assert a.kind == "hedge"
    assert a.doc_ids == ("kb-062",)
    # strip legitimate dates/doc-ids before checking for phone-shaped digits
    scrubbed = re.sub(r"\d{4}-\d{2}-\d{2}|kb-\d+", "", a.answer)
    assert not re.search(r"\+?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", scrubbed), \
        "phone-number-shaped text in answer"
    assert "does not directly cover" in a.answer
