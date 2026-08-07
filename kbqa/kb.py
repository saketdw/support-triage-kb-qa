"""Knowledge-base loading, validation, and temporal version resolution.

The KB is a set of markdown documents with YAML front matter. Versioned
policies form *families* linked by ``supersedes`` / ``superseded_by``, each
version carrying an ``[effective_date, valid_until]`` window. Two rules drive
everything downstream:

- **Eligibility is decided by dates alone.** The ``status`` field describes
  *today* (a doc can be ``superseded`` now yet be the correct citation for a
  historical question), so it is validated but never used for resolution.
- **Fail loudly at load, degrade gracefully at query time.** A malformed KB
  raises ``KBError`` before it can produce a wrong answer; a date with no
  policy in force simply resolves to ``None``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

REQUIRED_FIELDS = {"doc_id", "title", "status", "effective_date"}
KNOWN_STATUS = {"current", "superseded", "expired"}


class KBError(ValueError):
    """A structural problem in the knowledge base itself."""


def _as_date(value, field_name: str, path: Path) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        raise KBError(f"{path.name}: {field_name} is not an ISO date: {value!r}")


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    category: str
    status: str
    effective_date: date
    valid_until: date | None
    supersedes: str | None
    superseded_by: str | None
    body: str
    path: str

    def in_force(self, as_of: date) -> bool:
        """Date containment only — deliberately ignores ``status`` (see module
        docstring)."""
        if self.effective_date and self.effective_date > as_of:
            return False
        if self.valid_until and self.valid_until < as_of:
            return False
        return True


def parse_doc(path: Path) -> Doc:
    raw = path.read_text(encoding="utf-8")
    parts = raw.split("---", 2)
    if len(parts) < 3:
        raise KBError(f"{path.name}: missing YAML front matter (expected --- ... --- body)")
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        raise KBError(f"{path.name}: unparseable front matter: {e}")
    missing = REQUIRED_FIELDS - {k for k, v in meta.items() if v not in (None, "")}
    if missing:
        raise KBError(f"{path.name}: missing required front matter fields: {sorted(missing)}")
    status = str(meta["status"]).strip()
    if status not in KNOWN_STATUS:
        raise KBError(f"{path.name}: unknown status {status!r} (expected one of {sorted(KNOWN_STATUS)})")

    eff = _as_date(meta.get("effective_date"), "effective_date", path)
    until = _as_date(meta.get("valid_until"), "valid_until", path)
    if eff and until and eff > until:
        raise KBError(f"{path.name}: effective_date {eff} is after valid_until {until}")

    clean = lambda v: (str(v).strip() or None) if v not in (None, "") else None
    return Doc(
        doc_id=str(meta["doc_id"]).strip(),
        title=str(meta["title"]).strip(),
        category=clean(meta.get("category")) or "uncategorized",
        status=status,
        effective_date=eff,
        valid_until=until,
        supersedes=clean(meta.get("supersedes")),
        superseded_by=clean(meta.get("superseded_by")),
        body=parts[2].strip(),
        path=str(path),
    )


@dataclass
class KnowledgeBase:
    docs: dict[str, Doc]
    families: dict[str, tuple[str, ...]] = field(default_factory=dict)  # doc_id -> ordered family

    @classmethod
    def load(cls, kb_dir: str | Path) -> "KnowledgeBase":
        kb_dir = Path(kb_dir)
        paths = sorted(kb_dir.glob("*.md"))
        if not paths:
            raise KBError(f"no knowledge-base documents found in {kb_dir}")
        docs: dict[str, Doc] = {}
        for p in paths:
            d = parse_doc(p)
            if d.doc_id in docs:
                raise KBError(f"duplicate doc_id {d.doc_id!r} ({p.name} and {Path(docs[d.doc_id].path).name})")
            docs[d.doc_id] = d
        kb = cls(docs=docs)
        kb._validate_links()
        kb._build_families()
        kb._validate_windows()
        return kb

    # ---------------------------------------------------------------- checks
    def _validate_links(self) -> None:
        for d in self.docs.values():
            for attr in ("supersedes", "superseded_by"):
                ref = getattr(d, attr)
                if ref and ref not in self.docs:
                    raise KBError(f"{d.doc_id}: {attr} points at nonexistent doc {ref!r}")
        for d in self.docs.values():
            if d.superseded_by and self.docs[d.superseded_by].supersedes != d.doc_id:
                raise KBError(f"asymmetric link: {d.doc_id} superseded_by {d.superseded_by}, "
                              f"but {d.superseded_by} does not supersede it")
            if d.supersedes and self.docs[d.supersedes].superseded_by != d.doc_id:
                raise KBError(f"asymmetric link: {d.doc_id} supersedes {d.supersedes}, "
                              f"but {d.supersedes} is not superseded_by it")

    def _build_families(self) -> None:
        seen: set[str] = set()
        for d in self.docs.values():
            if d.doc_id in seen:
                continue
            # walk to the head of the chain, then forward to the tail
            head = d
            while head.supersedes:
                head = self.docs[head.supersedes]
            chain = [head.doc_id]
            cur = head
            while cur.superseded_by:
                cur = self.docs[cur.superseded_by]
                if cur.doc_id in chain:
                    raise KBError(f"supersedes cycle involving {cur.doc_id}")
                chain.append(cur.doc_id)
            for doc_id in chain:
                self.families[doc_id] = tuple(chain)
                seen.add(doc_id)

    def _validate_windows(self) -> None:
        """Within a family, validity windows must not overlap: at most one
        version of a policy may be in force on any given date."""
        for chain in {tuple(v) for v in self.families.values()}:
            ordered = sorted((self.docs[i] for i in chain), key=lambda d: d.effective_date)
            for prev, nxt in zip(ordered, ordered[1:]):
                if prev.valid_until is None or prev.valid_until >= nxt.effective_date:
                    raise KBError(
                        f"overlapping validity windows in family {chain}: "
                        f"{prev.doc_id} (until {prev.valid_until}) vs {nxt.doc_id} (from {nxt.effective_date})"
                    )

    # ---------------------------------------------------------------- queries
    def resolve(self, doc_id: str, as_of: date) -> Doc | None:
        """Map ANY version of a policy to the sibling in force at ``as_of``.

        Returns ``None`` when the family had no version in force on that date
        (e.g. an expired notice) — the caller decides what that means.
        """
        for sibling in self.families.get(doc_id, (doc_id,)):
            d = self.docs[sibling]
            if d.in_force(as_of):
                return d
        return None

    def body_text(self, doc_id: str) -> str:
        """Document body with the markdown heading stripped."""
        return re.sub(r"^#.*\n", "", self.docs[doc_id].body).strip()
