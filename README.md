# Meridian Talent Ranker 🎯

> Ranks a 100,000-candidate pool for a Senior AI Engineer role the way a great recruiter would —
> by **understanding the role, not matching keywords**.

A hybrid (dense + lexical + structured) candidate-ranking system for an **Intelligent Candidate
Discovery & Ranking** challenge. It reads the job description into structured
requirements, scores every candidate with complementary signals fused into one explainable number,
and produces a validated top-100 — **CPU-only, no GPU, no network at ranking time, ~55s for 100K**.

```bash
# 1. install (Python 3.12)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (one-time, offline) build the embedding cache — ~15s for 100K on CPU
python precompute.py --candidates ./candidates.jsonl --artifacts ./artifacts

# 3. produce the submission — the single reproduce command, ~55s, CPU-only, no network
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 4. validate against the official validator
python tools/validate_submission.py submission.csv      # -> "Submission is valid."
```

Step 2 builds the embedding cache (`artifacts/`, ~15s, one-time); it is intentionally **not**
committed to keep the repo lean, so run it before step 3. Without it, `rank.py` still runs and
degrades gracefully to the structural + lexical ranker.

---

## The key insight

The JD *tells you* the answer, and even warns: *"find candidates whose skills section contains the
most AI keywords … is a trap we've explicitly built into the dataset."* We verified it: the pool
contains **~4,300 keyword-stuffer traps** (non-tech people — HR Managers, Marketing Managers —
listing 6+ AI skills) and **~80 honeypots** (internally impossible profiles). The provided
`sample_submission.csv` deliberately ranks the stuffers at the top — it's the naive baseline.

A great recruiter reasons about the **gap between what the JD says and what it means**:

| The JD says… | …what it means | How we encode it |
|---|---|---|
| "embeddings / vector DB / ranking eval" | hard requirements | trust-weighted must-have coverage |
| "a Tier-5 may not say 'RAG' but *built a recommender*" | reward real work, not buzzwords | embed the **career narrative**, not the skills bag |
| "all AI keywords but title is Marketing Manager → not a fit" | catch the trap | role/career grounding + skill-trust + a stuffer penalty |
| "inactive 6 months, 5% response rate → not actually hireable" | availability matters | multiplicative behavioural modifier |
| disqualifiers (consulting-only, pure-research, title-chaser…) | hard negatives | per-signal penalties |
| ~80 honeypots, forced to tier 0 | don't get fooled | consistency validator crushes them |

---

## How it works

### 1. JD understanding (`src/jd.py`)
The single released JD is encoded as a structured `JobRequirements`: the 4 **must-haves**, the
**ideal profile** (6–8y, applied ML at *product* companies, NLP/IR, Noida/Pune), 8 **disqualifiers**,
location/experience/notice logic, and a distilled **dense "role-meaning" query** used for semantic
matching.

### 2. Feature extraction (`src/features.py`)
One streaming pass turns each profile into ~25 auditable signals: title fit, **career-history domain
evidence** (retrieval/ranking/recsys), must-have coverage, **skill-trust** (endorsements × duration ×
assessment), product-vs-services, NLP-vs-CV, experience-band fit, education, location bucket,
**behavioural availability**, and disqualifier flags.

### 3. Dense similarity (`src/embeddings.py`)
We embed each candidate's **career narrative** (headline + summary + role descriptions — *not* the
skills array) with **model2vec `potion-retrieval-32M`**, a static distilled embedding model: 100K
profiles in ~13s on CPU, no GPU. Embeddings are pre-computed offline into a 51 MB int8 cache; the
ranking step loads them with numpy only.

> Embedding the *narrative* rather than the *skill bag* is the decisive anti-trap design: a
> keyword-stuffer's narrative is about marketing/HR, so it scores low — while a SWE whose description
> says "built a recommendation engine serving millions" scores high without any buzzwords.

### 4. Hybrid scoring (`src/scoring.py`)

```
base  = Σ wᵢ · componentᵢ
        title .18 · domain-evidence .26 · must-haves .20 · semantic .16
        experience .08 · nice-to-haves .07 · education .05

score = base × location_modifier        (Noida/Pune > NCR metros > India > abroad)
             × behavioural_modifier       (active & responsive vs dormant)
             × disqualifier_multiplier     (consulting-only, title-chaser, CV-only, stuffer…)
             × consistency_multiplier       (honeypots → ×0.03)
```

Additive "fit" modulated by multiplicative "reachability & trust" — a transparent,
learning-to-rank-style blend chosen deliberately over a per-candidate LLM call (which can't meet the
CPU / 5-minute / no-network budget over 100K candidates).

### 5. Grounded reasoning (`src/reasoning.py`)
Every rank ships a 1–2 sentence justification assembled **only from the candidate's real fields**.
There is no free-text generation step, so a claim can't exist unless the underlying field does —
**hallucination-proof by construction**. Tone scales with rank (consistency); >90% of reasonings are
unique (variation). This directly targets the Stage-4 reasoning checks.

### 6. Consistency / honeypot validation (`src/validation.py`)
Flags unambiguous internal contradictions (tenure > calendar window; a single role longer than the
whole career; "expert" in 5+ skills with 0 months used). Flagged profiles are crushed to the floor,
so they cannot reach the top-100.

---

## Results

Generated by `python eval/evaluate.py` (full report in [`docs/RESULTS.md`](docs/RESULTS.md)):

| Metric | Our top-100 | Naive baseline (sample_submission) |
|---|---|---|
| Honeypots in top-100 | **0** (Stage-3 DQ is >10) | — |
| Keyword-stuffers / non-tech in top-10 | **0** | **6** of 10 |
| In ideal 5–9y band | **81/100** (median 6.4y) | — |
| Based in India | **99/100** | — |
| Real retrieval/ranking career evidence | **99/100** | — |
| Active (≤90d) & responsive | **100/100** | — |

Archetype tests assert the core contract: **real-builder > keyword-stuffer > honeypot**, behavioural
dormancy costs, consulting-only / CV-only / abroad-static are penalised.

---

## Compute compliance

| Constraint | Limit | Us |
|---|---|---|
| Runtime (ranking step) | ≤ 5 min | **~55s** for 100K |
| Memory | ≤ 16 GB | **< 2 GB** |
| Compute | CPU only | CPU only (no GPU anywhere) |
| Network | off | rank.py makes **no** network calls |
| Disk (intermediate) | ≤ 5 GB | 51 MB embedding cache |

Pre-computation (embeddings) is a separate ~15s offline step, as permitted by the spec.

---

## Repository layout

```
rank.py                 # the reproduce command (CPU, offline, numpy-only)
precompute.py           # offline embedding cache builder (model2vec)
src/
  jd.py                 # structured JD understanding
  features.py           # profile -> ~25 auditable signals
  scoring.py            # hybrid score = Σ wᵢ·componentᵢ × modifiers
  embeddings.py         # model2vec compute path + numpy cache path
  reasoning.py          # grounded, varied, rank-consistent reasoning
  validation.py         # honeypot / consistency detection
  config.py             # weights + lexicons (all tunables in one place)
  schema.py, text.py    # I/O + lexicon matching
artifacts/              # embedding cache — not committed; built by precompute.py (~15s)
app/streamlit_app.py    # interactive demo + hosted sandbox
app/demo_candidates.jsonl   # 340-candidate demo mix (fits + traps + honeypots)
eval/evaluate.py        # composition audit + baseline contrast
tests/                  # archetype / format / reasoning tests (pytest)
tools/validate_submission.py   # official validator (vendored)
submission.csv          # the top-100 ranking
```

## Tests

```bash
pytest -q          # 19 tests: archetype ordering, official-format, reasoning quality
```

## Demo / sandbox

```bash
streamlit run app/streamlit_app.py
```
Loads a bundled 340-candidate mix (genuine fits + keyword-stuffer traps + honeypots) and ranks them
live in ~2s, with a full per-candidate breakdown — and shows the traps/honeypots sinking to the
bottom. Deployable as-is to Streamlit Community Cloud (the required hosted sandbox).

## Notes
- The 100K `candidates.jsonl` is provided by the organisers and is **not** committed; point
  `--candidates` at your local copy.
- No LLM is in the ranking path — `rank.py` is numpy + standard library only.
