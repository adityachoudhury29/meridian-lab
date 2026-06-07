#!/usr/bin/env python3
"""Produce the top-100 ranking CSV from the candidate pool.

This is the single reproduce command (Stage-3 reproduction target). It runs
CPU-only, makes no network calls, and completes well within the 5-minute /
16 GB budget over the full 100K pool. The dense embedding model is NOT loaded
here — we consume the pre-computed artifacts produced by ``precompute.py``.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

If the embedding artifacts are absent, the ranker degrades gracefully to its
structural + lexical components (still a valid, if slightly weaker, ranking),
so the pipeline never hard-fails.
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from src import embeddings as E
from src.features import extract_features
from src.jd import build_jd
from src.reasoning import reasoning_for
from src.schema import iter_candidates, reference_date
from src.scoring import normalized_output_scores, rank, score_pool


def _reference_date(path: str):
    """One cheap pass to fix the deterministic 'now' (max last_active_date)."""
    mx = ""
    for c in iter_candidates(path):
        la = c.get("redrob_signals", {}).get("last_active_date", "")
        if la > mx:
            mx = la
    return reference_date([{"redrob_signals": {"last_active_date": mx}}])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="path to candidates.jsonl[.gz]")
    ap.add_argument("--out", default="./submission.csv")
    ap.add_argument("--artifacts", default="./artifacts")
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    jd = build_jd()
    ref = _reference_date(args.candidates)

    # --- feature extraction (one streaming pass; raw dicts are not retained) ---
    feats = []
    for c in iter_candidates(args.candidates):
        feats.append(extract_features(c, ref, jd))
    n = len(feats)
    print(f"[rank] extracted features for {n:,} candidates in {time.time()-t0:.1f}s (ref={ref})")

    # --- dense semantic component (from cache; graceful fallback) ---
    semantic = np.zeros(n, dtype=np.float32)
    if E.cache_exists(args.artifacts):
        index, cand_emb, jd_emb = E.load_cache(args.artifacts)
        rows = np.array([index.get(f.candidate_id, -1) for f in feats])
        have = rows >= 0
        aligned = np.zeros((n, cand_emb.shape[1]), dtype=np.float32)
        aligned[have] = cand_emb[rows[have]]
        raw = E.semantic_scores(aligned, jd_emb)
        raw[~have] = 0.0
        semantic = E.rescale(raw)
        print(f"[rank] applied dense embeddings for {int(have.sum()):,}/{n:,} candidates")
    else:
        print("[rank] WARNING: no embedding artifacts found — using structural+lexical only "
              "(run precompute.py for the full hybrid score)")

    # --- score, rank, reason ---
    scored = score_pool(feats, semantic)
    top = rank(scored, top_k=args.top_k)
    out_scores = normalized_output_scores(top)

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, s in enumerate(top):
            w.writerow([s.candidate_id, i + 1, f"{out_scores[i]:.6f}", reasoning_for(s, i + 1)])

    hp = sum(1 for s in top if s.features.is_honeypot)
    print(f"[rank] wrote {len(top)} rows to {args.out} in {time.time()-t0:.1f}s total "
          f"(honeypots in top-{args.top_k}: {hp})")


if __name__ == "__main__":
    main()
