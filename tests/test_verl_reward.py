from __future__ import annotations

import json

import pytest

from arlab.train.verl_reward import compute_score, extract_json, score_structured


def answer_gt(answer: str = "400") -> str:
    return json.dumps({"answer": answer})


def structured_gt() -> str:
    return json.dumps(
        {
            "service": "growth-feed",
            "status": "incident",
            "error_rate": 0.42,
            "citations": ["ev-1", "ev-2"],
        }
    )


def test_protocol_reply_still_scores():
    """A model that answers in the tool protocol is not penalised for the wrapper.

    ``answers_match`` falls back to the last number in the reply, so
    ``{"final_answer": "400"}`` still counts as 400.
    """
    out = compute_score("math", '```json\n{"final_answer": "400"}\n```', answer_gt())
    assert out["score"] == pytest.approx(1.0)


def test_answer_match_plain_text():
    out = compute_score("math", "结果是 400", answer_gt())
    assert out["score"] == pytest.approx(1.0)
    assert out["acc"] == pytest.approx(1.0)


def test_answer_mismatch_is_zero():
    out = compute_score("math", "42", answer_gt())
    assert out["score"] == pytest.approx(0.0)
    assert out["acc"] == pytest.approx(0.0)


def test_structured_partial_credit():
    reply = json.dumps(
        {
            "service": "growth-feed",   # correct            -> 1.0
            "status": "healthy",        # wrong              -> 0.0
            "error_rate": 0.4205,       # within 1%          -> 1.0
            "citations": ["ev-1"],      # 1 of 2 expected    -> 0.5
        }
    )
    out = compute_score("diag", reply, structured_gt())
    assert out["score"] == pytest.approx((1.0 + 0.0 + 1.0 + 0.5) / 4)
    assert out["acc"] == pytest.approx(0.0)


def test_numeric_tolerance_is_one_percent():
    """0.43 vs 0.42 is outside the 1% band, so that field earns nothing."""
    reply = json.dumps({"service": "growth-feed", "status": "incident", "error_rate": 0.43})
    out = compute_score("diag", reply, structured_gt())  # 4 expected fields, 3 replied
    assert out["score"] == pytest.approx((1.0 + 1.0 + 0.0 + 0.0) / 4)


def test_structured_requires_json():
    out = compute_score("diag", "the service looks fine", structured_gt())
    assert out["score"] == pytest.approx(0.0)


def test_malformed_ground_truth_is_reported_not_raised():
    out = compute_score("x", "anything", "not json")
    assert out["score"] == 0.0
    assert out["malformed_ground_truth"] == pytest.approx(1.0)


def test_empty_ground_truth_object_is_zero():
    out = compute_score("x", "anything", "{}")
    assert out["score"] == 0.0


def test_extra_info_is_optional_and_ignored():
    out = compute_score("math", "400", answer_gt(), None, some_other_kwarg=1)
    assert out["score"] == pytest.approx(1.0)


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('prefix {"b": 2} suffix') == {"b": 2}
    assert extract_json("no json") is None
    assert extract_json("") is None
    assert extract_json("[1, 2]") is None


def test_score_structured_ignores_extra_fields():
    expected = {"service": "growth-feed"}
    reply = json.dumps({"service": "growth-feed", "extra": "ignored"})
    assert score_structured(reply, expected) == pytest.approx(1.0)
