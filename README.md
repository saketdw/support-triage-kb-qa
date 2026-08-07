# Support Triage & KB Question Answering

![tests](https://github.com/saketdw/support-triage-kb-qa/actions/workflows/ci.yml/badge.svg)

Take-home practical in two parts:

- **Part A** — review of an inherited support-ticket route classifier: verdict,
  the honest evaluation, and a minimal working fix.
- **Part B** — question answering over a *versioned* knowledge base, where an
  answer that is right today can be wrong for the date being asked about.

Design stance throughout: **the simplest tool that meets the requirement,
measured, with a named trigger for each upgrade.** Every number below is
computed in this repo (the EDA notebook or `eval/`), not asserted.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # scikit-learn, numpy, PyYAML
```

**Entry point 1 — route classification** (input: CSV with a `text` column):

```bash
python -m triage.predict --input messages.csv --output predictions.csv
```

Output columns: `text,prediction`. Optional `--review-threshold 0.7` adds a
`needs_review` column (selective prediction; off by default so the schema
matches the brief exactly).

**Entry point 2 — question answering** (input: CSV with `qid,question,as_of`):

```bash
python -m kbqa.batch --input questions.csv --output answers.csv
```

Output columns: `qid,answer,doc_ids` (semicolon-separated; empty on
abstention). A missing `as_of` assumes today (with a warning); a malformed one
fails loudly with the row number.

**Tests and evaluation:**

```bash
pip install -r requirements-dev.txt
pytest -q                 # 49 tests (2 skip without tesseract)
python -m eval.evaluate   # retrieval + abstention metrics (writes eval/results.md)
```

---

## Part A — do I sign off? **No — and not for the obvious reason.**

The baseline reports **98.75%** and recommends shipping. I reproduced it and
audited the protocol before trusting it
(full evidence: [notebooks/01_eda_baseline_review.ipynb](notebooks/01_eda_baseline_review.ipynb)).

**What is wrong with how it is evaluated:**

1. **Preprocessing leakage** — the TF-IDF vectorizer is fitted on all 400 rows
   *before* the train/test split, so the evaluation saw the test set.
   (Measured: 96 vocabulary entries exist only because of test rows; idf drift
   is third-decimal.)
2. **Duplicate leakage** — the corpus is generated from ~169 body templates
   wrapped in `{greeting} + {body} + {asset} + {closing}`; **303 of 400 rows
   sit in a multi-row template family**. A random split therefore scores the
   model on phrasings it has effectively memorized. (Cross-checked
   independently by cosine-similarity clustering.)
3. **No uncertainty** — one seed, n=80: the Wilson 95% CI for 79/80 is
   **[93.3%, 99.8%]**. The reported precision is noise-level.
4. **Wrong metric for the stakes** — accuracy on a 40%-majority dataset hides
   the 12.5% class that is the costliest to get wrong. No per-class metrics,
   no confusion matrix in the report.
5. **`predict()` retrains per call** — and trains on all 400 rows, deploying a
   different model than the one evaluated.

**The honest number is a range, because the protocol decides the question.**

| Protocol | The question it answers | Accuracy | Fraud recall |
|---|---|---|---|
| As shipped (leaky, one 80-row split) | — | 98.75% | not reported |
| Stratified 5-fold, pipeline | traffic resembles this corpus | 100% | **1.00** |
| **Template-grouped 5-fold** | an entire phrasing family is new | **93.75%** | **0.76** |
| *the gap* | *the memorization component* | *6.2 pp* | *24 pp* |

So the reported figure is **both unsupported** (invalid protocol; CI [93.3%,
99.8%]) **and optimistic** about genuinely new phrasings: under family-level
grouping the model misses roughly **one fraud ticket in four** (Wilson 95% CI
[0.63, 0.86]).

**Why — and what the failure actually is** (notebook §5, §7, §8, §9). The model
reads surface vocabulary, not intent: benign text containing fraud words is
flagged (5 of 6 hand-written probes, one at 0.948 confidence), and fraud
phrased against the grain is missed. The obvious worry is that an adversarial
class drifts out from under it, so I tested that — and it mostly **does not**
hold. The discriminative axis is **agency**, not mechanism: `unauthorized`,
`stolen`, `gone`, `permission` appear in fraud tickets and in **0%** of dispute
tickets, and agency language reads the same in 2024 as in 2026. Four of five
scam types that post-date this dataset (address poisoning, pig-butchering,
malicious dApps, caller-ID spoofing) route correctly with no matching training
vocabulary.

Two things survive that test, and they matter more than drift:

- **Novelty shows up as lost confidence before lost correctness.** Mean
  confidence falls from **0.82** on familiar mechanisms to **0.45** on novel
  ones — and *every* novel probe, including the one misrouted (a token-approval
  drainer, at 0.35), lands under the review band. The band doesn't need to know
  which scams are new, only that the model is unsure.
- **The residual confusion is permanent, not drifting.** *"My balance
  disappeared overnight **and I want it back**"* routes to disputes despite
  unambiguous fraud agency, because remedy phrasing is dispute-coded (`want`
  appears in 24% of dispute tickets and 0% of fraud ones). No amount of fresh
  scam data fixes that.

I would quote **fraud recall 0.76–1.00** to a PM and say plainly that where a
given month lands depends on how close incoming *phrasing* is to phrasing
already seen — not, per §9, on whether the scams themselves are new. A point
estimate here would be false precision.

**The metric in production:** fraud-report recall with a floor, measured on
real traffic (never on this fixture), at acceptable per-class precision, plus
drift monitoring on input text and confidence distributions, with agent
re-routes harvested as free labels.

**What makes 0.76 shippable is not a better model but selective prediction.**
No alternative is significantly better (McNemar; three — NB-on-TF-IDF
p=0.0001, NB-on-counts p=0.013, gradient boosting p<0.0001 — are significantly
*worse*). But confidence separates cleanly: median **0.876 when right vs 0.518
when wrong**, so a `conf < 0.70` band routes **22% of tickets to a human and
catches 25/25 errors including all 12 missed frauds**. Ship the model *with*
the band — that is what `--review-threshold` exists for. Ship gate: a shadow
pilot on real traffic, because nothing measured on a synthetic corpus
transfers by default.

**The fix** ([baseline/baseline_classifier_fixed.py](baseline/baseline_classifier_fixed.py),
a small diff as invited): vectorizer inside a `Pipeline` (leakage becomes
structurally impossible), both CV protocols reported with per-class metrics and
the review band, `class_weight='balanced'` (kept on cost grounds — it is not
statistically distinguishable), `min_df=2` (removes all 387 single-document
memorization hooks for ~0.5 pp accuracy, fraud recall unchanged), fit-once
serving. The CLI (`triage/`) wraps the same reviewed config — a test asserts
they cannot drift apart.

---

## Part B — answering from a versioned KB

### The trap in the data, and the architecture it forces

The obvious design — *filter documents to those valid at `as_of`, then rank* —
scores **55%**. The reason: every current version has been rewritten away from
the customer's vocabulary:

| The customer says | v1 said | The current version says |
|---|---|---|
| "withdrawal fee" | Withdrawal Fees | *Network and Transfer Charges* (never "fee") |
| "fraud" | Reporting Fraud | *Reporting Unauthorized Activity* (never "fraud") |
| "staking" | Staking Rewards Rates | *Earn Rates* (never "staking") |
| "daily limit" | Daily Account Limits | *Verification Tiers and Transfer Ceilings* |

So retrieval runs over **all versions of all documents** — an old version
matching the customer's words identifies the right *topic family* — and the
`supersedes` chain then resolves to the version in force at `as_of`
(`kbqa/kb.py`). Old versions are not noise; **they are the index into the
family.** Measured effect: hit@1 on answerable questions **46% → 88%**.

Version selection is therefore **structural, not probabilistic**: eligibility
is decided by date containment alone (`status` describes *today* and is
deliberately ignored for historical questions), and a test enforces the
invariant that no answer can ever cite a document that was not in force at the
question's date. The same mechanism answers expired notices for free: a family
with no member in force resolves to nothing, and the system says *"that
maintenance window ended 2026-06-14"* instead of quoting the dead notice.

### Answers are extractive, by design

Answer text is verbatim sentences from the resolved document, always prefixed
`As of <date> (<doc>, in force <window>)` — so a historical answer can never
masquerade as current policy. Extraction makes hallucination impossible by
construction, which in regulated support is the property that matters: when
asked for a fraud-team phone number that the KB never publishes, the system
answers with the actual reporting channel and *says the number is not
published* — it cannot invent one, and a test proves it.

Four answer paths, in decision order: **abstain** (top retrieval score below
threshold → "escalating to a human agent") · **negative** (nothing in force →
what ended and when, citing the expired doc as evidence) · **hedge** (right
document, missing detail → name the gap, quote what the doc does say) ·
**answer** (extracted sentences, up to two documents when a second family
genuinely contributes).

### The honest evaluation

Gold labels ([eval/gold.csv](eval/gold.csv)) were written by reading the KB
*before* looking at system output; three-valued (`doc_id` /
`NEGATIVE:evidence` / `NONE`); evaluation-only — the answering code never sees
them. Full method and tables: [eval/results.md](eval/results.md). Summary:

| Bucket | n | hit@1 | hit@3 | Wilson 95% CI |
|---|---|---|---|---|
| current policy | 10 | 10/10 | 10/10 | [72%, 100%] |
| historical (superseded versions) | 4 | 4/4 | 4/4 | **[51%, 100%]** |
| active notices | 2 | 1/2 | 2/2 | [9%, 91%] |
| evergreen | 10 | 8/10 | 9/10 | [49%, 94%] |
| **overall answerable** | **26** | **23/26 (88%)** | | [71%, 96%] |

Expired notices: **4/4** answered as negatives with evidence. Unanswerable
bucket (n=8): 2 abstained, 1 hedged safely (the phone-number trap), **5
answered a related-but-wrong document** — the score distributions of
answerable and unanswerable questions overlap (measured: 10 of 26 answerable
questions score below the highest unanswerable one), so a raw threshold cannot
separate them; the sweep in `eval/results.md` shows the coverage/risk curve,
and the fix I would pursue is coverage checking (does the retrieved document
actually contain the entity type asked about), not a magic number.

Stated plainly: **these buckets are small.** The bucket that tests the
exercise's central concern (historical) is n=4; its perfect score is evidence
the mechanism is wired correctly, not a performance claim. The KB's own
metadata can generate hundreds of dated questions with known correct documents
("what was X on date D?" for every version window) — the first thing I would
build with more time.

The three retrieval misses are all *semantic*, not temporal: "Dogecoin" vs
"DOGE" (abbreviation), a current notice losing to a topical doc on word
overlap (q19), and "withdrawal address" hijacked by the fee family (q24).
Those are the measured trigger for an embedding re-ranker — deliberately not
taken at 31 documents.

### At scale (the 10k requests/minute question)

167 req/s. Routing and retrieval are deterministic CPU paths in microseconds —
they never need an LLM. Only answer *synthesis* could use one; support traffic
is Zipfian, so synthesized answers would be cached keyed by **(normalized
question, resolved doc-version set)** — when a document changes, its version
changes the key, and stale entries die by construction. If the LLM is down or
over budget, the system degrades to the extractive answer, which is always
available. Staleness in production is caught three ways: this repo's golden
tests re-run on every KB change (canary suite); an alarm on any answer citing
a superseded/expired doc for a current-date query (structurally impossible →
any alert means the loader broke); drift monitoring on retrieval scores plus
the human re-route rate.

---

## Considered and deliberately not built

| Option | Why not, with the measurement |
|---|---|
| LLM answer synthesis | Not required by the problem: extraction already guarantees groundedness. Would be a cacheable *rewriter* on top of the same citations (design above); adds a key dependency the brief asks to avoid. |
| Embeddings / vector DB | The most defensible upgrade on the table — §5/§8 diagnose a *vocabulary* problem, and embeddings are the tool for it. Deferred, not dismissed: no alternative reaches significance in the favourable direction (McNemar), 31 KB docs fit in one matrix, and a model download taxes every reviewer. Trigger: measured paraphrase misses on real traffic. |
| Fine-tuned router | 400 rows from ~169 templates; nothing measurable to gain (power: ~1,200 samples per arm to prove a 2 pp lift), and per-call cost/latency at 167 rps. |
| Chunking, rerankers, agent frameworks | 31 short documents. The complexity budget went to temporal correctness, which is what the exercise says fails in production. |

## Part C (second modality) — screenshots via local OCR

Built as an optional extra (`brew install tesseract && pip install -r
requirements-partc.txt`, then):

```bash
python -m partc.ocr_route --input media/screenshots --output routed.csv
```

**Architecture: one router, many input adapters.** The screenshot is OCR'd
locally and the text goes through the *same* classifier as chat and email —
no second model, no new training data; every router improvement benefits
every channel. Measured on the three provided images:

| file | OCR conf | route | route conf | needs_review |
|---|---|---|---|---|
| login-error.png | 91.6 | account-access ✓ | 0.95 | false |
| phishing-sms.png | 93.2 | fraud-report ✓ | **0.31** | **true** |
| txn-failed.png | 93.0 | transaction-dispute ✓ | 0.96 | false |

Three correct routes — and the middle row is the interesting one: the
phishing SMS routes *correctly but at 0.31 confidence*. That is **covariate
shift** made visible — OCR output (UI chrome + an SMS thread) is not the
distribution the classifier was trained on — and it is caught by the review
band, not by OCR confidence (which was a happy 93). The two gates are
deliberately different instruments: OCR confidence catches bad *images*, the
route-confidence band catches bad *fits*. (Honesty note: n=3 synthetic
renders demonstrates the wiring, not performance. Dark-mode inversion is
applied by luminance check; on these clean renders it measured neutral —
Tesseract 5 copes — and is kept as free insurance for low-contrast real
captures.)

**Why local, not a hosted vision API:** these images contain an email
address, a **live one-time code**, a wallet address and a $4,500 balance.
Shipping them to a third-party API is a data-processing event and deposits an
active credential in someone else's request logs. Local OCR costs ~$0 and
~200 ms/ticket vs ~$0.001–0.01 and 1–3 s hosted. In production, derived text
is redacted (OTP-shaped digit runs, wallet addresses) before touching any
log, and the pixels are dropped after triage. **What fails first live:** OCR
quality on real-world images — photos of screens, crops, messaging-app
recompression — gated here by per-word confidence + word count with fallback
to human triage, and monitored via the downstream classifier's confidence on
OCR'd vs native traffic (the adapter can degrade even when OCR reports itself
happy).

---

## Scope & trade-offs

**Prioritized:** temporal correctness as a structural guarantee (the stated
central failure mode); honest measurement with uncertainty everywhere
(both parts); tests that encode invariants, not examples; input validation on
both entry points.

**Deliberately left out:** anything in the table above; a persisted model
artifact (training takes ~1 s, so train-at-startup is simpler and cannot go
stale); packaging/Docker (two documented commands, stdlib + 3 deps).

**With more time:** auto-generate a dated evaluation set from the KB's own
version windows (hundreds of gold questions for free); coverage-based
abstention; calibrate the classifier unweighted + explicit threshold from a
cost matrix; the cached LLM rewriter behind a flag.

**Honest time spent:** about 8 focused hours end to end — roughly half on
analysis, evaluation design and labeling, half on implementation, tests and
write-up. (The brief's 3-hour budget shaped what was cut — everything in the
"not built" table — but I chose to spend real time understanding the data
first; the vocabulary-drift finding that drives the whole Part B architecture
came from that reading.)

**AI assistance:** built with AI-assisted tooling throughout, as the brief
assumes. Every change entered the repo through a pull request I reviewed and
merged, and every decision in this README is one I can defend without the
tool.

---

## Repo map

```
baseline/            the inherited files (verbatim) + baseline_classifier_fixed.py (the reviewed fix)
triage/              Part A entry point: model config + predict CLI
kbqa/                Part B: kb.py (load/validate/resolve) · retrieval.py · answer.py · batch.py
eval/                gold labels, evaluation harness, results.md
notebooks/           01_eda_baseline_review.ipynb — executed evidence log for Part A
partc/               optional screenshot adapter: local OCR into the same router
tests/               46 tests incl. a hermetic mini-KB reproducing the drift trap
examples/            sample inputs and the outputs the CLIs produce on them
data/ kb/ media/ questions.csv   starter assets, unmodified
```
