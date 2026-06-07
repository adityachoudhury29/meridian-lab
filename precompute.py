#!/usr/bin/env python3
"""Offline pre-computation of dense embedding artifacts.

This is the ONLY step that touches the embedding model / network, and it is
explicitly allowed to exceed the 5-minute ranking budget. It produces, under
``artifacts/``:

    candidate_ids.npy      pool-order candidate ids
    candidate_emb_f16.npy  (N, 384) L2-normalised float16 candidate embeddings
    jd_emb.npy             (1+k, 384) JD primary + aspect query embeddings

``rank.py`` consumes these with numpy only — no model, no network — so the
reproduced ranking step satisfies the CPU-only / offline / <5 min constraints.

Usage:
    python precompute.py --candidates ./candidates.jsonl --artifacts ./artifacts
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from src import embeddings as E
from src.jd import build_jd
from src.schema import career_narrative, iter_candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--artifacts", default="./artifacts")
    ap.add_argument("--model", default=E.MODEL_NAME)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-chars", type=int, default=1200,
                    help="truncate each narrative to keep embedding fast (role-essence is at the front)")
    args = ap.parse_args()

    t0 = time.time()
    ids: list[str] = []
    texts: list[str] = []
    for c in iter_candidates(args.candidates):
        ids.append(c["candidate_id"])
        texts.append(career_narrative(c)[: args.max_chars])
    print(f"[precompute] loaded {len(ids):,} candidate narratives in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    cand_emb = E.embed_texts(texts, model_name=args.model, batch_size=args.batch_size, progress=True)
    print(f"[precompute] embedded candidates {cand_emb.shape} in {time.time()-t1:.1f}s")

    jd = build_jd()
    jd_emb = E.build_jd_matrix(jd, model_name=args.model)
    print(f"[precompute] embedded JD matrix {jd_emb.shape}")

    E.save_cache(args.artifacts, ids, cand_emb, jd_emb)
    mb = cand_emb.astype(np.float16).nbytes / 1e6
    print(f"[precompute] saved artifacts to {args.artifacts} (candidate matrix ~{mb:.0f} MB) "
          f"in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
