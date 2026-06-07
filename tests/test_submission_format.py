"""Submission-format tests using the *official* validator shipped with the
hackathon bundle (vendored verbatim at tools/validate_submission.py).

These guard against the "common rejections" the spec lists (99/101 rows, ranks
from 0, duplicate ids, increasing scores, …) so we can never fail Stage 1.
"""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "official_validator", ROOT / "tools" / "validate_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VAL = _load_validator()


def _write(tmp_path, rows, name="team_test.csv"):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        w.writerows(rows)
    return p


def _good_rows():
    return [[f"CAND_{i:07d}", i, round(1.0 - i * 0.005, 4), f"reason {i}"]
            for i in range(1, 101)]


def test_valid_submission_passes(tmp_path):
    p = _write(tmp_path, _good_rows())
    assert VAL.validate_submission(str(p)) == []


def test_too_few_rows_rejected(tmp_path):
    p = _write(tmp_path, _good_rows()[:99])
    assert VAL.validate_submission(str(p))


def test_increasing_score_rejected(tmp_path):
    rows = [[f"CAND_{i:07d}", i, round(i * 0.005, 4), "r"] for i in range(1, 101)]
    assert VAL.validate_submission(str(p := _write(tmp_path, rows)))


def test_duplicate_candidate_rejected(tmp_path):
    rows = _good_rows()
    rows[1][0] = rows[0][0]
    assert VAL.validate_submission(str(_write(tmp_path, rows)))


def test_real_submission_is_valid():
    """If a submission has been generated, it must validate clean."""
    for name in ("submission.csv", "submission_structural.csv"):
        p = ROOT / name
        if p.exists():
            assert VAL.validate_submission(str(p)) == [], f"{name} failed official validator"
