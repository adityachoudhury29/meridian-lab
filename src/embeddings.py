"""Dense embedding component.

Split into two paths so the *ranking step* never needs the model or the network:

* compute path (``embed_texts`` / ``build_jd_matrix``) — used only offline in
  ``precompute.py`` and live in the Streamlit demo. Uses **model2vec** static
  distilled embeddings (``potion-retrieval-32M``): a transformer distilled down
  to a static token-embedding lookup, so encoding 100K profiles takes seconds on
  CPU with no GPU and no per-token forward pass. This is what lets the whole
  pipeline — even pre-computation — stay CPU-only and fast.

* cache path (``load_cache`` / ``semantic_scores``) — used by ``rank.py``. Pure
  ``numpy``: loads the pre-computed candidate matrix (stored int8-quantised to
  stay well under repo file-size limits) and the JD matrix. No model, no network.

All vectors are L2-normalised so cosine similarity is a plain dot product.
"""
from __future__ import annotations

import os

import numpy as np

MODEL_NAME = "minishlab/potion-retrieval-32M"
EMB_DIM = 512
_QSCALE = 127.0   # int8 quantisation scale for unit vectors

CAND_IDS_FILE = "candidate_ids.npy"
CAND_EMB_FILE = "candidate_emb_i8.npy"
JD_EMB_FILE = "jd_emb.npy"


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# --------------------------------------------------------------------------- #
# compute path (offline only)
# --------------------------------------------------------------------------- #
_MODEL = None


def _load_model(model_name: str = MODEL_NAME):
    global _MODEL
    if _MODEL is None:
        from model2vec import StaticModel  # local import keeps rank.py model-free
        _MODEL = StaticModel.from_pretrained(model_name)
    return _MODEL


def embed_texts(texts: list[str], model_name: str = MODEL_NAME, batch_size: int = 4096,
                progress: bool = False) -> np.ndarray:
    """Embed texts -> L2-normalised float32 (N, EMB_DIM)."""
    model = _load_model(model_name)
    out = []
    n = len(texts)
    for i in range(0, n, batch_size):
        out.append(np.asarray(model.encode(texts[i:i + batch_size]), dtype=np.float32))
        if progress and (i // batch_size) % 5 == 0:
            print(f"    …embedded {min(i + batch_size, n):,}/{n:,}", flush=True)
    mat = np.vstack(out) if out else np.zeros((0, EMB_DIM), dtype=np.float32)
    return _l2(mat)


def build_jd_matrix(jd, model_name: str = MODEL_NAME) -> np.ndarray:
    """JD matrix: row 0 = primary 'what the role means' query, rows 1.. = aspects."""
    queries = [jd.dense_query] + list(jd.dense_aspect_queries)
    return embed_texts(queries, model_name=model_name)


# --------------------------------------------------------------------------- #
# cache path (used by rank.py — numpy only)
# --------------------------------------------------------------------------- #
def save_cache(artifacts_dir: str, ids: list[str], cand_emb: np.ndarray, jd_emb: np.ndarray) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    np.save(os.path.join(artifacts_dir, CAND_IDS_FILE), np.asarray(ids))
    q = np.clip(np.round(cand_emb * _QSCALE), -127, 127).astype(np.int8)   # unit vecs -> int8
    np.save(os.path.join(artifacts_dir, CAND_EMB_FILE), q)
    np.save(os.path.join(artifacts_dir, JD_EMB_FILE), jd_emb.astype(np.float32))


def cache_exists(artifacts_dir: str) -> bool:
    return all(os.path.exists(os.path.join(artifacts_dir, f))
               for f in (CAND_IDS_FILE, CAND_EMB_FILE, JD_EMB_FILE))


def load_cache(artifacts_dir: str) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Return ``(id->row index, candidate_emb float32 renormalised, jd_emb float32)``."""
    ids = np.load(os.path.join(artifacts_dir, CAND_IDS_FILE), allow_pickle=True)
    cand = _l2(np.load(os.path.join(artifacts_dir, CAND_EMB_FILE)).astype(np.float32) / _QSCALE)
    jd = np.load(os.path.join(artifacts_dir, JD_EMB_FILE)).astype(np.float32)
    index = {str(cid): i for i, cid in enumerate(ids)}
    return index, cand, jd


def semantic_scores(cand_emb: np.ndarray, jd_emb: np.ndarray,
                    main_weight: float = 0.6, aspect_weight: float = 0.4) -> np.ndarray:
    """Per-candidate raw semantic similarity in roughly [-1, 1].

    Combines the primary query (weighted) with the best-matching aspect query, so
    a candidate who matches *any* facet of the role (retrieval, recsys, vectorDB,
    eval) is rewarded — mirroring how a recruiter reads partial-but-real fit.
    """
    main = cand_emb @ jd_emb[0]
    if jd_emb.shape[0] > 1:
        best_aspect = (cand_emb @ jd_emb[1:].T).max(axis=1)
    else:
        best_aspect = main
    return main_weight * main + aspect_weight * best_aspect


def rescale(sim: np.ndarray, lo: float = 0.18, hi: float = 0.62) -> np.ndarray:
    """Map raw model2vec cosine-ish similarity to [0,1] via a calibrated stretch.
    Calibrated to the potion-retrieval-32M scale (lower-magnitude cosines than a
    full transformer)."""
    return np.clip((sim - lo) / (hi - lo), 0.0, 1.0)
