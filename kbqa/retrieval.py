"""Lexical retrieval over the knowledge base: retrieve first, resolve second.

The obvious design — filter documents to those valid at ``as_of``, then rank —
fails on this KB, because current versions were rewritten away from the
customer's vocabulary (measured in the evaluation: the current fee document
never says "withdrawal" or "fee"; the current fraud document never says
"fraud"). Superseded versions still speak the customer's language.

So the index deliberately spans **all versions of all documents**: an old
version matching the question identifies the right *topic family*, and
:meth:`kbqa.kb.KnowledgeBase.resolve` then walks the family to the version in
force at ``as_of``. Old versions are not noise — they are the index into the
family. This took hit@1 on the answerable questions from 46% to 88%.

Plain TF-IDF cosine, document level, no chunking: 31 short documents with
distinctive vocabulary do not justify anything heavier. The measured residual
failures (see eval/) are semantic — the trigger to consider an embedding
re-ranker, deliberately not taken at this scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kbqa.kb import Doc, KnowledgeBase


@dataclass(frozen=True)
class ResolvedHit:
    """One retrieval hit after temporal resolution."""
    doc: Doc | None          # version in force at as_of; None = family has none
    matched_id: str          # the version the query text actually matched
    score: float             # cosine similarity of the matched version


class Retriever:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self._ids = sorted(kb.docs)
        # Title is the strongest topical signal in short policy docs; weight it
        # by repetition (title x2 + body).
        corpus = [
            f"{kb.docs[i].title} {kb.docs[i].title} {kb.body_text(i)}" for i in self._ids
        ]
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._matrix = self._vec.fit_transform(corpus)

    def rank(self, question: str) -> list[tuple[str, float]]:
        """All documents (every version), best match first."""
        sims = cosine_similarity(self._vec.transform([question]), self._matrix)[0]
        order = np.argsort(sims)[::-1]
        return [(self._ids[i], float(sims[i])) for i in order]

    def retrieve(self, question: str, as_of: date, k: int = 3) -> list[ResolvedHit]:
        """Top-k hits, resolved to the version in force at ``as_of``.

        Deduplicates by family (the first — highest-scoring — version of a
        family wins), preserving score order. A hit whose family has no
        version in force at ``as_of`` is kept with ``doc=None``: for notices
        this *is* the answer ("that window has ended"), not a failure.
        """
        hits: list[ResolvedHit] = []
        seen_families: set[tuple[str, ...]] = set()
        for doc_id, score in self.rank(question):
            family = self.kb.families.get(doc_id, (doc_id,))
            if family in seen_families:
                continue
            seen_families.add(family)
            hits.append(ResolvedHit(doc=self.kb.resolve(doc_id, as_of), matched_id=doc_id, score=score))
            if len(hits) >= k:
                break
        return hits
