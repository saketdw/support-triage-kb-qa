"""Retrieval and abstention evaluation against eval/gold.csv.

    python -m eval.evaluate

Reports per bucket (the buckets are the point of the exercise — an aggregate
would hide that the temporal bucket behaves differently from the evergreen
one):

- answerable buckets: hit@1, hit@3, MRR, and a Wilson 95% CI on hit@1 —
  stated because the buckets are small (n=4 for historical: a perfect score
  there is evidence the mechanism works, not a performance claim);
- expired-notice bucket: counted correct only when the system answers a
  *negative* citing the expired document as evidence;
- unanswerable bucket: abstain = correct, an explicit "not covered" hedge =
  safe (extraction cannot invent facts), a plain answer = wrong;
- an abstention-threshold sweep, because the score distributions of
  answerable and unanswerable questions overlap — the sweep shows the
  coverage/risk trade-off the fixed threshold picks from.
"""
from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path

from kbqa.answer import AnswerService

ROOT = Path(__file__).resolve().parents[1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def main() -> None:
    gold = {r["qid"]: r for r in csv.DictReader(open(ROOT / "eval" / "gold.csv"))}
    questions = {r["qid"]: r for r in csv.DictReader(open(ROOT / "questions.csv"))}
    assert set(gold) == set(questions), "gold.csv and questions.csv disagree on qids"

    service = AnswerService(ROOT / "kb")
    lines: list[str] = []
    say = lambda s="": (print(s), lines.append(s))

    per_bucket: dict[str, list[dict]] = {}
    for qid, g in gold.items():
        q = questions[qid]
        as_of = date.fromisoformat(q["as_of"])
        result = service.answer(q["question"], as_of)
        # top-3 resolved doc ids for hit@3 / MRR
        top3 = [h.doc.doc_id for h in service.retriever.retrieve(q["question"], as_of, k=3) if h.doc]
        per_bucket.setdefault(g["bucket"], []).append(
            {"qid": qid, "expected": g["expected"], "result": result, "top3": top3}
        )

    say("## Retrieval — answerable buckets\n")
    say("| bucket | n | hit@1 | hit@3 | MRR | Wilson 95% CI on hit@1 |")
    say("|---|---|---|---|---|---|")
    total_k = total_n = 0
    for bucket in ["current", "historical", "active-notice", "evergreen"]:
        rows = per_bucket[bucket]
        k1 = sum(r["result"].doc_ids[:1] == (r["expected"],) for r in rows)
        k3 = sum(r["expected"] in r["top3"] for r in rows)
        mrr = sum(
            1.0 / (r["top3"].index(r["expected"]) + 1) if r["expected"] in r["top3"] else 0.0
            for r in rows
        ) / len(rows)
        lo, hi = wilson(k1, len(rows))
        total_k += k1
        total_n += len(rows)
        say(f"| {bucket} | {len(rows)} | {k1}/{len(rows)} | {k3}/{len(rows)} | {mrr:.2f} | [{lo:.0%}, {hi:.0%}] |")
    lo, hi = wilson(total_k, total_n)
    say(f"| **overall** | {total_n} | **{total_k}/{total_n} ({total_k/total_n:.0%})** | | | [{lo:.0%}, {hi:.0%}] |")

    say("\n## Expired notices (answerable as a negative)\n")
    rows = per_bucket["expired-notice"]
    ok = sum(
        r["result"].kind == "negative" and r["result"].doc_ids == (r["expected"].split(":", 1)[1],)
        for r in rows
    )
    say(f"{ok}/{len(rows)} answered as a negative citing the expired document as evidence.")

    say("\n## Unanswerable bucket (gold = NONE)\n")
    rows = per_bucket["unanswerable"]
    counts = {"abstain": [], "hedge": [], "negative": [], "answer": []}
    for r in rows:
        counts[r["result"].kind].append(r["qid"])
    say("| behavior | count | qids | reading |")
    say("|---|---|---|---|")
    say(f"| abstained (correct) | {len(counts['abstain'])} | {', '.join(counts['abstain'])} | escalated to a human |")
    say(f"| hedged (safe) | {len(counts['hedge'])} | {', '.join(counts['hedge'])} | named the gap; extraction cannot invent facts |")
    say(f"| answered (wrong) | {len(counts['answer']) + len(counts['negative'])} | "
        f"{', '.join(counts['answer'] + counts['negative'])} | quoted a related but non-answering document |")

    say("\n## Abstention threshold sweep\n")
    say("The fixed threshold (0.10) is one point on this curve; the sweep is the honest picture.\n")
    say("| threshold | abstained | of which truly unanswerable | answerable lost |")
    say("|---|---|---|---|")
    unanswerable = {r["qid"] for r in rows}
    flat = [r for rs in per_bucket.values() for r in rs]
    for tau in [0.05, 0.10, 0.15, 0.20, 0.25]:
        abstained = [r for r in flat if r["result"].score < tau]
        tp = sum(r["qid"] in unanswerable for r in abstained)
        say(f"| {tau:.2f} | {len(abstained)} | {tp} | {len(abstained) - tp} |")

    say("\n*Buckets are small; every number above carries the interval shown or wider. "
        "The historical bucket (n=4) at 4/4 is consistent with a true rate as low as 51% — "
        "the KB's own metadata could generate hundreds more dated questions to tighten it "
        "(see README, 'with more time').*")

    out = ROOT / "eval" / "results.md"
    out.write_text("# Evaluation results\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
