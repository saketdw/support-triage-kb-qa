"""KB loading, integrity validation, and temporal resolution."""
from datetime import date
from pathlib import Path

import pytest

from kbqa.kb import KBError, KnowledgeBase

MINI = Path(__file__).parent / "fixtures" / "kb_mini"

HEADER = """---
doc_id: {doc_id}
title: {title}
category: test
version: 1
effective_date: {eff}
valid_until: {until}
status: {status}
supersedes: {supersedes}
superseded_by: {superseded_by}
---

# {title}

Body text long enough to be split into sentences later on.
"""


def write_doc(dirpath, doc_id, eff, until="", status="current", supersedes="", superseded_by="", title=None):
    (dirpath / f"{doc_id}.md").write_text(
        HEADER.format(doc_id=doc_id, title=title or doc_id, eff=eff, until=until,
                      status=status, supersedes=supersedes, superseded_by=superseded_by),
        encoding="utf-8",
    )


# ---------------------------------------------------------------- happy path
def test_loads_mini_kb_and_builds_families():
    kb = KnowledgeBase.load(MINI)
    assert set(kb.docs) == {"fam-001", "fam-002", "notice-001", "evergreen-001"}
    assert kb.families["fam-001"] == ("fam-001", "fam-002")
    assert kb.families["fam-002"] == ("fam-001", "fam-002")


def test_resolution_is_by_date_from_any_version():
    kb = KnowledgeBase.load(MINI)
    # historical date -> v1, regardless of which version we start from
    assert kb.resolve("fam-002", date(2024, 6, 1)).doc_id == "fam-001"
    # current date -> v2, even starting from the superseded doc
    assert kb.resolve("fam-001", date(2025, 6, 1)).doc_id == "fam-002"


def test_status_field_is_ignored_for_resolution():
    """A doc that is 'superseded' TODAY was current in its window: it must
    still resolve for historical dates. status describes today, not as_of."""
    kb = KnowledgeBase.load(MINI)
    v1 = kb.docs["fam-001"]
    assert v1.status == "superseded"
    assert kb.resolve("fam-001", date(2024, 6, 1)).doc_id == "fam-001"


def test_expired_notice_resolves_to_none():
    kb = KnowledgeBase.load(MINI)
    assert kb.resolve("notice-001", date(2025, 3, 2)).doc_id == "notice-001"  # during window
    assert kb.resolve("notice-001", date(2025, 6, 1)) is None                 # after window


# ---------------------------------------------------------------- integrity
def test_rejects_dangling_link(tmp_path):
    write_doc(tmp_path, "a-001", "2024-01-01", superseded_by="ghost-999", status="superseded", until="2024-12-31")
    with pytest.raises(KBError, match="nonexistent"):
        KnowledgeBase.load(tmp_path)


def test_rejects_asymmetric_links(tmp_path):
    write_doc(tmp_path, "a-001", "2024-01-01", until="2024-12-31", status="superseded", superseded_by="a-002")
    write_doc(tmp_path, "a-002", "2025-01-01")  # missing supersedes: a-001
    with pytest.raises(KBError, match="asymmetric"):
        KnowledgeBase.load(tmp_path)


def test_rejects_overlapping_windows(tmp_path):
    """At most one version of a policy may be in force on any date."""
    write_doc(tmp_path, "a-001", "2024-01-01", until="2025-06-30", status="superseded", superseded_by="a-002")
    write_doc(tmp_path, "a-002", "2025-01-01", supersedes="a-001")
    with pytest.raises(KBError, match="overlapping"):
        KnowledgeBase.load(tmp_path)


def test_rejects_inverted_dates(tmp_path):
    write_doc(tmp_path, "a-001", "2025-12-31", until="2025-01-01")
    with pytest.raises(KBError, match="after valid_until"):
        KnowledgeBase.load(tmp_path)


def test_rejects_malformed_date(tmp_path):
    write_doc(tmp_path, "a-001", "not-a-date")
    with pytest.raises(KBError, match="ISO date"):
        KnowledgeBase.load(tmp_path)


def test_rejects_missing_required_field(tmp_path):
    (tmp_path / "a-001.md").write_text(
        "---\ndoc_id: a-001\nstatus: current\n---\n\n# t\n\nbody\n", encoding="utf-8"
    )
    with pytest.raises(KBError, match="missing required"):
        KnowledgeBase.load(tmp_path)
