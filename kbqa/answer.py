"""Extractive question answering with temporal grounding and abstention.

Answer generation is **extractive by design**: the answer text is verbatim
sentences from the resolved document, prefixed with the policy version and its
validity window. Every word is traceable to a citation, so hallucination is
impossible by construction — in regulated support, a confidently wrong answer
costs far more than an escalation. (An LLM rewriter would be a cacheable
polish layer on top of this exact output; deliberately not built at this
scope.)

Four answer paths, in decision order:

1. **Abstain** — top retrieval score below ``ABSTAIN_THRESHOLD``. Measured on
   the provided questions, score distributions of answerable and unanswerable
   questions overlap heavily, so the threshold is deliberately conservative
   and abstention is phrased as an escalation, not a dead end.
2. **Nothing in force** — the best-matching family has no version valid at
   ``as_of`` (an expired notice/promotion). This is answerable as a
   *negative*: state what ended and when, citing the expired document as
   evidence.
3. **Low sentence coverage** — the right document rarely contains the exact
   fact asked for (e.g. a phone number the KB never publishes). Answer with
   what the document does say, explicitly flagging that the specific detail
   is not covered — never invent it.
4. **Extractive answer** — top sentences by lexical match, re-ordered to
   document order, prefixed with "As of <date> (<doc> ..., in force ...)" so a
   historical answer can never masquerade as current policy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kbqa.kb import Doc, KnowledgeBase
from kbqa.retrieval import Retriever

# Tuned on the provided questions.csv (stated plainly: n=38 leaves no held-out
# slice; the evaluation reports the full coverage/risk sweep instead).
ABSTAIN_THRESHOLD = 0.10   # top cosine below this -> escalate to a human
SECOND_DOC_MIN = 0.15      # a 2nd document must clear this absolute score...
SECOND_DOC_RATIO = 0.75    # ...and 75% of the top score, to be included
# The missing-detail hedge fires only on essentially-zero sentence coverage.
# Moderate-but-low coverage is expected under version vocabulary drift (the
# customer says "fee"; the current doc says "charges") and quotes normally,
# because family-level retrieval already established topicality.
SENTENCE_MIN = 0.02
MAX_SENTENCES = 2

ABSTAIN_TEXT = (
    "This question is not covered by the knowledge base, so I will not guess - "
    "escalating to a human agent."
)


@dataclass(frozen=True)
class Answer:
    answer: str
    doc_ids: tuple[str, ...]
    kind: str                      # "answer" | "hedge" | "negative" | "abstain"
    score: float                   # top retrieval score (abstention evidence)
    sentences: tuple[str, ...] = field(default=())


def split_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if len(s.strip()) > 15]


def _window(doc: Doc) -> str:
    until = doc.valid_until.isoformat() if doc.valid_until else "present"
    return f"in force {doc.effective_date.isoformat()} to {until}"


def _prefix(doc: Doc, as_of: date) -> str:
    """Every answer names its version and window, so a historical answer can
    never read as current policy ('Current rates are...' from a 2025 doc)."""
    return f"As of {as_of.isoformat()} ({doc.doc_id} \"{doc.title}\", {_window(doc)}): "


class AnswerService:
    def __init__(self, kb_dir: str | Path = None):
        kb_dir = kb_dir or Path(__file__).resolve().parents[1] / "kb"
        self.kb = KnowledgeBase.load(kb_dir)
        self.retriever = Retriever(self.kb)

    # ------------------------------------------------------------- internals
    def _extract(self, question: str, doc: Doc) -> tuple[list[str], float]:
        sentences = split_sentences(self.kb.body_text(doc.doc_id))
        if not sentences:
            return [], 0.0
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        try:
            matrix = vec.fit_transform(sentences + [question])
        except ValueError:  # question is all stop words
            return sentences[:MAX_SENTENCES], 0.0
        sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
        best = float(sims.max())
        if best < SENTENCE_MIN:
            # Zero lexical coverage: similarity ordering is arbitrary noise.
            # Policy documents lead with the operative rule, so fall back to
            # the document-initial sentences instead of a random pick.
            return sentences[:MAX_SENTENCES], best
        picked = sorted(np.argsort(sims)[::-1][:MAX_SENTENCES])  # back to document order
        return [sentences[i] for i in picked], best

    # ------------------------------------------------------------- public
    def answer(self, question: str, as_of: date | str) -> Answer:
        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)
        question = (question or "").strip()
        if not question:
            raise ValueError("question is empty")

        hits = self.retriever.retrieve(question, as_of, k=3)
        top = hits[0] if hits else None
        if top is None or top.score < ABSTAIN_THRESHOLD:
            return Answer(ABSTAIN_TEXT, (), "abstain", top.score if top else 0.0)

        # Best-matching family has no version in force at as_of: a negative
        # answer with the expired document as evidence (q: "is the promo still
        # running?" -> "no, it ended on ...").
        if top.doc is None:
            other = self.kb.docs[top.matched_id]
            first_line = " ".join(split_sentences(self.kb.body_text(other.doc_id))[:1])
            if other.effective_date and other.effective_date > as_of:
                # Asked about a date before this policy family existed. Name the
                # family's *earliest* version, not whichever one happened to match.
                family = self.kb.families.get(other.doc_id, (other.doc_id,))
                earliest = min((self.kb.docs[d] for d in family), key=lambda d: d.effective_date)
                first_line = " ".join(split_sentences(self.kb.body_text(earliest.doc_id))[:1])
                text = (
                    f"No policy on this was in force on {as_of.isoformat()}: the earliest "
                    f"relevant document, \"{earliest.title}\" ({earliest.doc_id}), only takes "
                    f"effect on {earliest.effective_date.isoformat()}. From that date it said: "
                    f"{first_line}"
                )
                return Answer(text, (earliest.doc_id,), "negative", top.score)

            text = (
                f"No current policy: \"{other.title}\" ({other.doc_id}) was "
                f"{_window(other)} and is not in force on {as_of.isoformat()}. "
                f"For reference, it said: {first_line}"
            )
            return Answer(text, (other.doc_id,), "negative", top.score)

        # Right document — but does it actually contain the asked-for detail?
        sentences, sent_score = self._extract(question, top.doc)
        prefix = _prefix(top.doc, as_of)
        if sent_score < SENTENCE_MIN or not sentences:
            text = (
                prefix
                + f"The knowledge base does not directly cover this specific detail. "
                  f"What the closest policy does say: {' '.join(sentences[:1]) if sentences else '(no relevant text)'}"
            )
            return Answer(text, (top.doc.doc_id,), "hedge", top.score, tuple(sentences[:1]))

        doc_ids = [top.doc.doc_id]
        body = " ".join(sentences)

        # A close-scoring second family can genuinely contribute (e.g. fraud
        # reporting + phishing guidance); include one extra sentence, cited.
        second = hits[1] if len(hits) > 1 else None
        if (
            second is not None
            and second.doc is not None
            and second.score >= max(SECOND_DOC_MIN, SECOND_DOC_RATIO * top.score)
        ):
            extra, extra_score = self._extract(question, second.doc)
            if extra and extra_score >= SENTENCE_MIN:
                body += f" Related ({second.doc.doc_id} \"{second.doc.title}\"): {extra[0]}"
                doc_ids.append(second.doc.doc_id)

        return Answer(prefix + body, tuple(doc_ids), "answer", top.score, tuple(sentences))


_DEFAULT: AnswerService | None = None


def answer(question: str, as_of: date | str) -> Answer:
    """The interface the exercise asks for: answer(question, as_of) ->
    answer text + the documents used. Uses the repo's kb/ directory."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = AnswerService()
    return _DEFAULT.answer(question, as_of)
