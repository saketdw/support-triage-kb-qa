"""Answering behavior on the hermetic mini-KB.

The fixture reproduces the real KB's central trap in miniature: the current
version (fam-002, "Gadget Transfer Charges") no longer uses the customer's
vocabulary from v1 ("Widget Fees"). These tests pin the behaviors that matter,
independent of the real KB's content.
"""
from datetime import date
from pathlib import Path

import pytest

from kbqa.answer import AnswerService

MINI = Path(__file__).parent / "fixtures" / "kb_mini"


@pytest.fixture(scope="module")
def svc():
    return AnswerService(MINI)


def test_temporal_resolution_historical(svc):
    a = svc.answer("What is the widget fee?", date(2024, 6, 1))
    assert a.doc_ids == ("fam-001",)
    assert "2%" in a.answer
    assert a.answer.startswith("As of 2024-06-01")


def test_temporal_resolution_current_despite_vocabulary_drift(svc):
    """The question uses v1 vocabulary ('widget fee'); the in-force doc says
    'gadget transfer charges'. Retrieve-then-resolve must still land on v2 —
    and must NOT leak the stale v1 facts into the answer."""
    a = svc.answer("What is the widget fee?", date(2025, 6, 1))
    assert a.doc_ids == ("fam-002",)
    assert "1%" in a.answer
    assert "2%" not in a.answer               # the superseded figure
    assert "minimum charge of $5" not in a.answer  # the stale fact, by name


def test_notice_during_window_is_quoted(svc):
    a = svc.answer("Are gadget transfers paused for downtime?", date(2025, 3, 2))
    assert a.kind in ("answer", "hedge")
    assert a.doc_ids == ("notice-001",)


def test_notice_after_window_is_a_negative_not_a_stale_quote(svc):
    a = svc.answer("Are gadget transfers paused for downtime?", date(2025, 6, 1))
    assert a.kind == "negative"
    assert a.doc_ids == ("notice-001",)  # cited as evidence
    assert "not in force" in a.answer
    assert "2025-03-05" in a.answer      # says when it ended


def test_out_of_scope_question_abstains(svc):
    a = svc.answer("What is the weather forecast for Paris tomorrow?", date(2025, 6, 1))
    assert a.kind == "abstain"
    assert a.doc_ids == ()
    assert "escalat" in a.answer.lower()


def test_citations_are_always_temporally_valid(svc):
    """The structural invariant: an answer may only cite documents in force at
    as_of — except a 'negative', whose whole point is citing the expired
    document as evidence."""
    probes = [
        ("What is the widget fee?", date(2024, 6, 1)),
        ("What is the widget fee?", date(2025, 6, 1)),
        ("What are the gadget charges?", date(2025, 6, 1)),
        ("When is support available?", date(2025, 6, 1)),
        ("Are gadget transfers paused?", date(2025, 6, 1)),
    ]
    for question, as_of in probes:
        a = svc.answer(question, as_of)
        for doc_id in a.doc_ids:
            doc = svc.kb.docs[doc_id]
            if a.kind == "negative":
                assert not doc.in_force(as_of)
            else:
                assert doc.in_force(as_of), f"{doc_id} cited but not in force at {as_of}"


def test_deterministic(svc):
    a1 = svc.answer("What is the widget fee?", date(2025, 6, 1))
    a2 = svc.answer("What is the widget fee?", date(2025, 6, 1))
    assert a1 == a2


def test_empty_question_rejected(svc):
    with pytest.raises(ValueError):
        svc.answer("   ", date(2025, 6, 1))
