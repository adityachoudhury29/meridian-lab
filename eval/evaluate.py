#!/usr/bin/env python3
"""Evaluation & sanity report.

There is no public ground truth, so we evaluate the *shape* of the ranking
against everything the JD asks for: title composition, location mix, experience
band, product-vs-services, must-have coverage, behavioural availability, and the
honeypot rate (the Stage-3 disqualifier). We also contrast our top-10 against the
provided naive baseline (sample_submission.csv) to show we avoid its
keyword-stuffer trap.

    python eval/evaluate.py --submission submission.csv \
        --candidates candidates.jsonl [--baseline sample_submission.csv] [--out docs/RESULTS.md]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import NONTECH_TITLES
from src.features import extract_features
from src.jd import build_jd
from src.schema import reference_date
from src.validation import consistency_check


def load_ids(csv_path):
    return [r["candidate_id"] for r in csv.DictReader(open(csv_path))]


def index_candidates(path, wanted):
    wanted = set(wanted)
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            if c["candidate_id"] in wanted:
                out[c["candidate_id"]] = c
    return out


def section(title):
    return f"\n## {title}\n"


def report(sub_ids, cand_by_id, ref, jd, baseline_ids=None):
    top = [cand_by_id[i] for i in sub_ids if i in cand_by_id]
    feats = [extract_features(c, ref, jd) for c in top]

    titles = Counter(c["profile"]["current_title"] for c in top)
    countries = Counter(c["profile"]["country"] for c in top)
    cities = Counter(c["profile"]["location"].split(",")[0] for c in top)
    inds = Counter(c["profile"]["current_industry"] for c in top)
    yoes = [c["profile"]["years_of_experience"] for c in top]

    nontech = sum(1 for c in top if c["profile"]["current_title"].lower() in NONTECH_TITLES)
    abroad = sum(1 for c in top if c["profile"]["country"] != "India")
    honeypots = sum(1 for c in top if consistency_check(c, ref)[0])
    in_band = sum(1 for y in yoes if 5 <= y <= 9)
    mh3 = sum(1 for f in feats if len(f.musthave_present) >= 3)
    domain = sum(1 for f in feats if f.domain_evidence >= 0.45)
    active = sum(1 for f in feats if f.last_active_days <= 90)
    responsive = sum(1 for f in feats if f.response_rate >= 0.4)

    L = []
    L.append("# Ranking quality report\n")
    L.append(f"- Top-N evaluated: **{len(top)}**")
    L.append(f"- **Honeypots in top-100: {honeypots}** "
             f"(Stage-3 disqualifies > 10; honeypot rate = {honeypots}%)")
    L.append(f"- Non-tech / keyword-stuffer titles in top-100: **{nontech}**")
    L.append(f"- In ideal 5–9y experience band: **{in_band}/{len(top)}** "
             f"(median {st.median(yoes):.1f}y, mean {st.mean(yoes):.1f}y)")
    L.append(f"- Based in India: **{len(top)-abroad}/{len(top)}** (abroad: {abroad})")
    L.append(f"- ≥3/4 hard requirements evidenced: **{mh3}/{len(top)}**")
    L.append(f"- Career-history domain evidence (retrieval/ranking/recsys): **{domain}/{len(top)}**")
    L.append(f"- Active in last 90 days: **{active}/{len(top)}** · responsive (≥40%): **{responsive}/{len(top)}**")

    L.append(section("Title composition (top-100)"))
    for t, n in titles.most_common():
        L.append(f"- {n:>3}  {t}")

    L.append(section("Location mix (top cities)"))
    L.append(", ".join(f"{c} ({n})" for c, n in cities.most_common(12)))
    L.append("\n\n**Countries:** " + ", ".join(f"{c} ({n})" for c, n in countries.most_common()))

    L.append(section("Industry mix"))
    L.append(", ".join(f"{c} ({n})" for c, n in inds.most_common(10)))

    if baseline_ids:
        b_top = [cand_by_id[i] for i in baseline_ids[:10] if i in cand_by_id]
        b_nontech = [c for c in b_top if c["profile"]["current_title"].lower() in NONTECH_TITLES]
        L.append(section("Contrast vs naive baseline (sample_submission.csv)"))
        L.append(f"- Baseline top-10 non-tech keyword-stuffers: **{len(b_nontech)}/10** "
                 f"→ e.g. " + "; ".join(f"{c['profile']['current_title']}"
                                        for c in b_nontech[:4]))
        L.append(f"- Our top-10 non-tech keyword-stuffers: "
                 f"**{sum(1 for c in top[:10] if c['profile']['current_title'].lower() in NONTECH_TITLES)}/10**")

    L.append(section("Top 10 (id · title · yrs · city)"))
    for i, c in enumerate(top[:10], 1):
        p = c["profile"]
        L.append(f"{i:>2}. {c['candidate_id']} · {p['current_title']} · "
                 f"{p['years_of_experience']:.1f}y · {p['location']}")

    baseline_nontech = 0
    if baseline_ids:
        b_top = [cand_by_id[i] for i in baseline_ids[:10] if i in cand_by_id]
        baseline_nontech = sum(1 for c in b_top
                               if c["profile"]["current_title"].lower() in NONTECH_TITLES)
    stats = {
        "honeypots": honeypots, "nontech": nontech, "baseline_nontech": baseline_nontech,
        "in_band": in_band, "median_yoe": st.median(yoes), "mean_yoe": st.mean(yoes),
        "india": len(top) - abroad, "domain": domain, "active": active,
        "mh3": mh3, "responsive": responsive, "n": len(top),
    }
    return "\n".join(L), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sub_ids = load_ids(args.submission)
    baseline_ids = load_ids(args.baseline) if args.baseline else None
    wanted = set(sub_ids) | set(baseline_ids or [])
    cand_by_id = index_candidates(args.candidates, wanted)
    ref = reference_date(list(cand_by_id.values()))
    text, stats = report(sub_ids, cand_by_id, ref, build_jd(), baseline_ids)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        stats_path = Path(args.out).with_name("results_stats.json")
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"\n[eval] wrote {args.out} and {stats_path}")


if __name__ == "__main__":
    main()
