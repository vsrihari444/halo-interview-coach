"""Rubric, prompts and automatic checks for the Halo Interview Coach.

The coach reads an interview question, the candidate's answer and (optionally) a
list of facts the candidate has confirmed about themselves, and returns ONE JSON
object of structured feedback. The hard rule: feedback never invents facts about
the candidate. Any suggested rewrite may only reuse facts present in the answer
or the confirmed facts; anything else must be a [bracketed placeholder].
"""
from __future__ import annotations

import json
import re
from typing import Any

DIMENSIONS = ("clarity", "specificity", "structure", "concept_coverage")

DIMENSION_GUIDE = {
    "clarity": "Is the answer easy to follow, direct, and free of filler or jargon misuse?",
    "specificity": "Does it give concrete actions, numbers, names of tools or decisions instead of generalities?",
    "structure": "Does it follow a sensible shape (e.g. STAR for behavioral, requirements→design→trade-offs for system design)?",
    "concept_coverage": "Does it cover the key concepts an interviewer expects for this question?",
}

SYSTEM_PROMPT = (
    "You are an interview coach. Evaluate the candidate's answer to the interview question. "
    "Score each dimension from 1 (poor) to 5 (excellent): "
    + "; ".join(f"{k}: {v}" for k, v in DIMENSION_GUIDE.items())
    + ". Rules: (1) Reply with ONE JSON object only, no prose around it. "
    "(2) Never invent facts about the candidate: no new employers, metrics, dates, tools or results. "
    "In rewrite_tip, reuse only facts stated in the answer or in confirmed_facts; write anything else as a "
    "[bracketed placeholder] the candidate fills in. (3) Be concrete and kind; every improvement must be actionable. "
    'JSON keys: "scores" {clarity, specificity, structure, concept_coverage: int 1-5}, '
    '"strengths" [str], "improvements" [str], "missing_concepts" [str], "rewrite_tip" str.'
)


def build_user_prompt(question: str, answer: str, confirmed_facts: list[str] | None = None, role: str | None = None) -> str:
    parts = []
    if role:
        parts.append(f"Target role: {role}")
    parts.append(f"Interview question: {question}")
    parts.append(f"Candidate answer: {answer}")
    facts = confirmed_facts or []
    parts.append("confirmed_facts: " + (json.dumps(facts) if facts else "[] (none provided)"))
    return "\n\n".join(parts)


def build_messages(ex: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(ex["question"], ex["answer"], ex.get("confirmed_facts"), ex.get("role"))},
    ]


# ---------------------------------------------------------------- parsing ---

def extract_json(text: str) -> dict[str, Any] | None:
    """Return the first JSON object in text, or None."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def schema_errors(fb: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    scores = fb.get("scores")
    if not isinstance(scores, dict):
        errs.append("scores missing")
    else:
        for d in DIMENSIONS:
            v = scores.get(d)
            if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 5:
                errs.append(f"scores.{d} must be int 1-5")
    for k in ("strengths", "improvements", "missing_concepts"):
        if not isinstance(fb.get(k), list) or not all(isinstance(x, str) for x in fb.get(k, [])):
            errs.append(f"{k} must be a list of strings")
    if not isinstance(fb.get("rewrite_tip"), str) or not fb.get("rewrite_tip", "").strip():
        errs.append("rewrite_tip must be a non-empty string")
    if isinstance(fb.get("improvements"), list) and len(fb["improvements"]) == 0:
        errs.append("improvements must not be empty")
    return errs


# ------------------------------------------------------ grounding check ---

_PLACEHOLDER = re.compile(r"\[[^\]]*\]")
_NUMBER = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?\s*(?:%|x|k|m|b|ms|s|hrs?|hours?|days?|weeks?|months?|years?)?", re.I)


def _numbers(text: str) -> set[str]:
    out = set()
    for m in _NUMBER.finditer(text):
        n = re.sub(r"[^\d.]", "", m.group(0))
        if n and n not in {"1", "2", "3", "4", "5"}:  # small ordinals/score values are not claims
            out.add(n.rstrip("."))
    return out


def fabricated_numbers(fb: dict[str, Any], answer: str, confirmed_facts: list[str] | None = None) -> list[str]:
    """Numbers in rewrite_tip that appear neither in the answer nor in the confirmed facts.

    Bracketed placeholders like "[your % improvement]" are ignored. A non-empty result
    means the coach likely invented a metric for the candidate.
    """
    tip = _PLACEHOLDER.sub(" ", str(fb.get("rewrite_tip", "")))
    source = answer + " " + " ".join(confirmed_facts or [])
    return sorted(_numbers(tip) - _numbers(source))


def auto_check(text: str, ex: dict[str, Any]) -> dict[str, Any]:
    """Automatic checks on one model output for one example."""
    fb = extract_json(text)
    if fb is None:
        return {"json_valid": False, "schema_valid": False, "grounded": False, "errors": ["no JSON object"]}
    errs = schema_errors(fb)
    fab = fabricated_numbers(fb, ex["answer"], ex.get("confirmed_facts"))
    res: dict[str, Any] = {
        "json_valid": True,
        "schema_valid": not errs,
        "grounded": not fab,
        "fabricated_numbers": fab,
        "errors": errs,
    }
    ref = ex.get("feedback") or {}
    if not errs and isinstance(ref.get("scores"), dict):
        diffs = [abs(fb["scores"][d] - ref["scores"][d]) for d in DIMENSIONS]
        res["score_mae_vs_reference"] = sum(diffs) / len(diffs)
    if not errs and ex.get("quality") in ("strong", "weak"):
        mean = sum(fb["scores"][d] for d in DIMENSIONS) / len(DIMENSIONS)
        res["calibrated"] = mean >= 3.5 if ex["quality"] == "strong" else mean <= 2.75
    return res
