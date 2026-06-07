"""Meridian Talent Ranker — interactive demo & hosted sandbox.

Doubles as the required sandbox (Streamlit Community Cloud, free tier): paste/edit
the role, load a candidate sample (the bundled 340-candidate mix or your own
JSONL), and watch the hybrid ranker separate genuine fits from keyword-stuffers
and honeypots — with a full, grounded explanation for every candidate.

The dense component embeds live with fastembed for the small sample (a few
seconds on CPU). If the model can't be fetched, the app degrades to the
structural+lexical ranker so the sandbox never hard-fails.

Run locally:   streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import embeddings as E          # noqa: E402
from src.features import extract_features  # noqa: E402
from src.jd import build_jd               # noqa: E402
from src.reasoning import reasoning_for   # noqa: E402
from src.schema import career_narrative, reference_date  # noqa: E402
from src.scoring import normalized_output_scores, rank, score_pool  # noqa: E402

st.set_page_config(page_title="Meridian Talent Ranker", page_icon="🎯", layout="wide")

PURPLE = "#6C4BF4"
CSS = f"""
<style>
.block-container {{ padding-top: 1.2rem; }}
.hero {{ background: linear-gradient(110deg, #1b1340 0%, {PURPLE} 55%, #2563eb 100%);
        color: white; padding: 18px 24px; border-radius: 14px; margin-bottom: 10px; }}
.hero h1 {{ margin: 0; font-size: 1.7rem; letter-spacing: .5px; }}
.hero p {{ margin: 4px 0 0; opacity: .9; }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.72rem;
        font-weight:600; margin-right:6px; }}
.fit {{ background:#dcfce7; color:#166534; }}
.trap {{ background:#fee2e2; color:#991b1b; }}
.hp {{ background:#fef9c3; color:#854d0e; }}
.small {{ color:#64748b; font-size:.8rem; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><h1>🎯 Meridian Talent Ranker</h1>'
    '<p>Ranks candidates the way a great recruiter would — reading the role, not matching keywords. '
    'Hybrid dense + lexical + structured scoring, CPU-only, fully explainable.</p></div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_demo() -> list[dict]:
    path = ROOT / "app" / "demo_candidates.jsonl"
    return [json.loads(l) for l in open(path) if l.strip()]


@st.cache_resource(show_spinner="Loading embedding model…")
def get_model():
    try:
        from model2vec import StaticModel
        return StaticModel.from_pretrained(E.MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        return exc


def embed(model, texts: list[str]) -> np.ndarray:
    mat = np.asarray(model.encode(texts), dtype=np.float32)
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1
    return mat / n


JD = build_jd()

# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("⚙️ Controls")
    source = st.radio("Candidate pool", ["Bundled demo sample (340)", "Upload JSONL"])
    candidates = load_demo()
    if source == "Upload JSONL":
        up = st.file_uploader("candidates .jsonl", type=["jsonl", "json"])
        if up is not None:
            candidates = [json.loads(l) for l in up.read().decode("utf-8").splitlines() if l.strip()]
    top_k = st.slider("Show top", 5, min(100, len(candidates)), 15)
    use_dense = st.toggle("Use dense embeddings (hybrid)", value=True,
                          help="Off = structural + lexical only")
    st.caption(f"Pool size: **{len(candidates)}** candidates")
    st.divider()
    st.markdown("**The 4 hard requirements**")
    for m in JD.must_haves:
        st.caption("• " + m.split("(")[0].strip())

st.markdown("#### 📋 Role intent (drives the dense match)")
intent = st.text_area("Edit the role's meaning — the ranker re-embeds it live:",
                      JD.dense_query, height=120)

run = st.button("🚀 Rank candidates", type="primary")

# --------------------------------------------------------------------------- #
if run:
    ref = reference_date(candidates)
    feats = [extract_features(c, ref, JD) for c in candidates]

    semantic = np.zeros(len(candidates), dtype=np.float32)
    dense_ok = False
    if use_dense:
        model = get_model()
        if isinstance(model, Exception):
            st.warning(f"Dense model unavailable ({model}); using structural+lexical only.")
        else:
            cand_emb = embed(model, [career_narrative(c) for c in candidates])
            jd_emb = embed(model, [intent] + list(JD.dense_aspect_queries))
            semantic = E.rescale(E.semantic_scores(cand_emb, jd_emb))
            dense_ok = True

    scored = score_pool(feats, semantic)
    ranked = rank(scored, top_k=len(candidates))
    out_scores = normalized_output_scores(ranked)
    by_id = {c["candidate_id"]: c for c in candidates}

    # pool-level "beyond keywords" insight
    from src.config import AI_CORE_SKILLS, NONTECH_TITLES
    n_traps = sum(1 for c in candidates
                  if c["profile"]["current_title"].lower() in NONTECH_TITLES
                  and sum(1 for s in c.get("skills", []) if s["name"].lower() in AI_CORE_SKILLS) >= 6)
    n_hp = sum(1 for f in feats if f.is_honeypot)
    trap_ranks = [i + 1 for i, s in enumerate(ranked)
                  if s.features.current_title.lower() in NONTECH_TITLES
                  and s.features.ai_core_skill_count >= 6]
    hp_ranks = [i + 1 for i, s in enumerate(ranked) if s.features.is_honeypot]

    c1, c2, c3 = st.columns(3)
    c1.metric("Keyword-stuffer traps in pool", n_traps,
              help="Non-tech titles listing ≥6 AI skills")
    c2.metric("Honeypots in pool", n_hp)
    c3.metric("Mode", "Hybrid (dense+lexical+structured)" if dense_ok else "Structural+lexical")
    worst = len(ranked)
    st.success(
        f"✅ Traps pushed to ranks { '/'.join(map(str, trap_ranks[:6])) or '—'} "
        f"(of {worst}); honeypots to {'/'.join(map(str, hp_ranks[:6])) or '—'}. "
        "Naive keyword matching would put these at the **top**."
    )

    st.markdown(f"### 🏅 Top {top_k}")
    for i, s in enumerate(ranked[:top_k], 1):
        c = by_id[s.candidate_id]
        p = c["profile"]
        f = s.features
        tag = ""
        if f.is_honeypot:
            tag = '<span class="pill hp">honeypot</span>'
        elif "keyword_stuffer_mismatch" in f.disqualifiers:
            tag = '<span class="pill trap">keyword-stuffer</span>'
        elif f.domain_evidence >= 0.5 and f.title_fit >= 0.7:
            tag = '<span class="pill fit">strong fit</span>'
        with st.expander(
            f"**#{i} · {p['current_title']} · {p['years_of_experience']:.1f}y · "
            f"{p['location']}**   —   score {out_scores[i-1]:.3f}", expanded=(i <= 3)):
            st.markdown(tag, unsafe_allow_html=True)
            st.markdown(f"**Reasoning:** {reasoning_for(s, i)}")
            bd1, bd2 = st.columns([3, 2])
            with bd1:
                st.markdown("**Score breakdown (weighted contributions)**")
                comp = dict(s.components)
                comp["semantic"] = comp.get("semantic", 0.0)
                st.bar_chart(comp, horizontal=True, height=240)
                st.caption(
                    f"× location {f.location_modifier:.2f} · × availability {f.behavioral_modifier:.2f}"
                    f" · × disqualifiers {f.disqualifier_multiplier:.2f}"
                    f" · × consistency {f.consistency_multiplier:.2f}")
            with bd2:
                if f.strengths:
                    st.markdown("**Strengths**")
                    for x in f.strengths[:5]:
                        st.markdown(f"- ✅ {x}")
                if f.concerns:
                    st.markdown("**Concerns (honest)**")
                    for x in f.concerns[:5]:
                        st.markdown(f"- ⚠️ {x}")
                st.caption(
                    f"last active {f.last_active_days}d ago · response {f.response_rate:.0%} · "
                    f"notice {f.notice_period_days}d · {f.ai_core_skill_count} AI skills listed")
else:
    st.info("Set your options in the sidebar and click **Rank candidates**. "
            "The bundled sample deliberately mixes genuine fits, keyword-stuffer traps and honeypots "
            "so you can see the ranker separate them.")
