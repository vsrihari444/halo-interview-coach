#!/usr/bin/env python3
"""Evaluate an interview-coach model: automatic checks + optional LLM judge.

Automatic checks (no API needed): JSON validity, schema validity, grounding (no invented numbers in
rewrite_tip), score MAE vs the reference feedback, and calibration (strong answers scored >= 3.5 on
average, weak answers <= 2.75).
LLM judge (optional, --judge-model): rates each feedback 1-10 on accuracy, actionability and
grounding against the reference, through any OpenAI-compatible API (default OpenRouter).

Prediction sources:
  --model M --base-url URL      query a served model (e.g. vLLM serving the Halo checkpoint)
  --predictions FILE            JSONL of {"id": ..., "output": "..."} produced elsewhere
  --reference-as-prediction     score the reference feedback itself (pipeline self-test, offline)

  python scripts/eval.py --data data/eval/test.jsonl --reference-as-prediction
  python scripts/eval.py --data data/eval/test.jsonl --model coach --base-url http://localhost:8000/v1 \
      --judge-model openai/gpt-oss-120b --out results/tuned.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from coach.rubric import auto_check, build_messages, build_user_prompt, extract_json  # noqa: E402
from scripts._llm import chat  # noqa: E402

JUDGE_PROMPT = """You grade feedback written by an interview coach. You see the interview question, the candidate's
answer, facts the candidate confirmed, a reference feedback written by an expert, and the coach's feedback.
Rate the coach's feedback from 1 to 10 on each criterion:
- accuracy: are the scores and observations correct for this answer (compare with the reference)?
- actionability: are the improvements concrete and useful?
- grounding: does it avoid inventing facts about the candidate (employers, metrics, results not in the answer or confirmed facts)? 10 = nothing invented.
Reply with JSON only: {"accuracy": int, "actionability": int, "grounding": int, "comment": str}"""


def judge(ex: dict, output: str, model: str, base_url: str | None) -> dict | None:
    content = (build_user_prompt(ex["question"], ex["answer"], ex.get("confirmed_facts"), ex.get("role"))
               + "\n\nReference feedback: " + json.dumps(ex.get("feedback"), ensure_ascii=False)
               + "\n\nCoach feedback: " + output)
    text = chat([{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": content}],
                model=model, base_url=base_url, temperature=0.0, max_tokens=800)
    j = extract_json(text)
    if not j:
        return None
    try:
        return {k: max(1, min(10, int(j[k]))) for k in ("accuracy", "actionability", "grounding")}
    except (KeyError, TypeError, ValueError):
        return None


def summarize(results: list[dict]) -> dict:
    n = len(results)
    s = {"n": n}
    for k in ("json_valid", "schema_valid", "grounded"):
        s[f"{k}_rate"] = round(sum(bool(r["auto"].get(k)) for r in results) / max(n, 1), 4)
    cal = [r["auto"]["calibrated"] for r in results if "calibrated" in r["auto"]]
    if cal:
        s["calibration_rate"] = round(sum(cal) / len(cal), 4)
    mae = [r["auto"]["score_mae_vs_reference"] for r in results if "score_mae_vs_reference" in r["auto"]]
    if mae:
        s["score_mae_vs_reference"] = round(statistics.mean(mae), 4)
    js = [r["judge"] for r in results if r.get("judge")]
    if js:
        for k in ("accuracy", "actionability", "grounding"):
            s[f"judge_{k}"] = round(statistics.mean(j[k] for j in js), 3)
        s["judge_n"] = len(js)
    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/eval/test.jsonl")
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--predictions")
    p.add_argument("--reference-as-prediction", action="store_true")
    p.add_argument("--judge-model")
    p.add_argument("--judge-base-url")
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--out")
    a = p.parse_args()

    data = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()][: a.limit]
    preds: dict[str, str] = {}
    if a.predictions:
        for l in open(a.predictions, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                preds[r["id"]] = r["output"]
    elif a.reference_as_prediction:
        preds = {ex["id"]: json.dumps(ex["feedback"], ensure_ascii=False) for ex in data}
    elif not a.model:
        sys.exit("Give --model (+ --base-url), --predictions, or --reference-as-prediction.")

    def run(ex: dict) -> dict:
        out = preds.get(ex["id"])
        if out is None:
            out = chat(build_messages(ex), model=a.model, base_url=a.base_url, temperature=0.0, max_tokens=a.max_tokens)
        rec = {"id": ex["id"], "output": out, "auto": auto_check(out, ex)}
        if a.judge_model:
            try:
                rec["judge"] = judge(ex, out, a.judge_model, a.judge_base_url)
            except Exception as e:  # judge outage should not kill the eval
                rec["judge_error"] = str(e)
        return rec

    with cf.ThreadPoolExecutor(a.workers) as pool:
        results = list(pool.map(run, data))
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
