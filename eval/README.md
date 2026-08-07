# Evaluation

`gold.csv` maps each provided question to the document a correct system should
cite **at that question's `as_of` date** — labeled by reading the KB, before
looking at system output. Three-valued:

- `kb-xxx` — the answer should cite this document.
- `NEGATIVE:kb-xxx` — answerable *as a negative* ("that notice/promo has
  ended"), with the expired document as evidence.
- `NONE` — not answerable from the KB; the correct behavior is to abstain
  (a hedged answer that explicitly names the gap is scored separately as
  *safe*, since extraction cannot invent facts).

Rules kept while labeling: the gold set is **evaluation-only** (the answering
code never reads it — that would be the Part A leakage mistake in a new
costume), and it is versioned with the KB snapshot it was labeled against.

Run:

    python -m eval.evaluate          # prints tables, writes eval/results.md
