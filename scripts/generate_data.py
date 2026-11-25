#!/usr/bin/env python3
"""Generate synthetic interview-feedback data through any OpenAI-compatible API (default: OpenRouter).

Stage 1 (writer model): invent a role, an interview question, a candidate answer of a planned
quality (strong / medium / weak) and, sometimes, a short list of facts the candidate has confirmed.
Stage 2 (teacher model): write reference feedback with the coach's system prompt, using seed
examples as few-shot demonstrations.
Every row is then checked automatically (JSON schema + grounding: no numbers in rewrite_tip that the
candidate never stated). Rows that fail are dropped, not repaired.

Examples:
  export OPENROUTER_API_KEY=...            # never commit this
  python scripts/generate_data.py --n 200 --out data/generated/batch1.jsonl
  python scripts/generate_data.py --n 3 --dry-run          # print prompts, no API calls
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from coach.rubric import SYSTEM_PROMPT, auto_check, build_user_prompt, extract_json  # noqa: E402
from scripts._llm import chat  # noqa: E402

ROLES = [
    "Software Engineer", "Backend Engineer", "Frontend Engineer", "Data Scientist", "Data Analyst",
    "Data Engineer", "Machine Learning Engineer", "DevOps Engineer", "Cloud Engineer", "Product Manager",
    "Engineering Manager", "Designer", "Account Executive", "Sales Development Representative",
    "Customer Success Manager", "Marketing Manager", "Project Manager", "QA Engineer", "Security Engineer",
]
CATEGORIES = ["behavioral", "system_design", "coding_concepts", "ml_concepts", "product_sense", "sales", "analytics", "infrastructure", "databases"]
QUALITY = ["strong", "medium", "weak"]

WRITER_PROMPT = """Write one realistic interview exchange as a JSON object with keys:
"role", "category", "question", "answer", "confirmed_facts".
Target role: {role}. Category: {category}. Planned answer quality: {quality}.
- strong: specific, structured (e.g. STAR), covers key concepts, 80-150 words.
- medium: on topic but vague in places or missing a key concept, 30-70 words.
- weak: generic, unstructured or partly wrong, 10-35 words.
confirmed_facts: {facts_rule}
Use invented but plausible details. Do not use real people's names. Reply with the JSON object only."""


def _facts_rule(rng: random.Random) -> str:
    if rng.random() < 0.35:
        return ("a list of 1-2 short facts the candidate confirmed in their profile (e.g. a metric or employer) that the "
                "answer FAILS to mention, so good feedback should tell them to use these facts")
    return "an empty list []"


def _few_shot(seed: list[dict], rng: random.Random, k: int) -> list[dict]:
    msgs = []
    for ex in rng.sample(seed, k=min(k, len(seed))):
        msgs.append({"role": "user", "content": build_user_prompt(ex["question"], ex["answer"], ex.get("confirmed_facts"), ex.get("role"))})
        msgs.append({"role": "assistant", "content": json.dumps(ex["feedback"], ensure_ascii=False)})
    return msgs


def make_one(i: int, args, seed: list[dict]) -> dict | None:
    rng = random.Random(args.seed * 100003 + i)
    role, category, quality = rng.choice(ROLES), rng.choice(CATEGORIES), rng.choice(QUALITY)
    wprompt = WRITER_PROMPT.format(role=role, category=category, quality=quality, facts_rule=_facts_rule(rng))
    tmsgs_prefix = [{"role": "system", "content": SYSTEM_PROMPT}] + _few_shot(seed, rng, args.shots)
    if args.dry_run:
        print(f"--- [{i}] writer prompt ---\n{wprompt}\n--- teacher: system + {args.shots} few-shot pairs ---", file=sys.stderr)
        return None
    ex = extract_json(chat([{"role": "user", "content": wprompt}], model=args.writer_model, temperature=1.0, json_mode=args.json_mode))
    if not ex or not all(isinstance(ex.get(k), str) and ex[k].strip() for k in ("question", "answer")):
        return None
    facts = ex.get("confirmed_facts") if isinstance(ex.get("confirmed_facts"), list) else []
    facts = [str(f) for f in facts][:3]
    row = {"role": ex.get("role") or role, "category": ex.get("category") or category, "quality": quality,
           "question": ex["question"].strip(), "answer": ex["answer"].strip(), "confirmed_facts": facts}
    user = build_user_prompt(row["question"], row["answer"], facts, row["role"])
    text = chat(tmsgs_prefix + [{"role": "user", "content": user}], model=args.teacher_model, temperature=0.3, json_mode=args.json_mode)
    check = auto_check(text, row)
    if not (check["schema_valid"] and check["grounded"]):
        return None
    if args.require_calibrated and check.get("calibrated") is False:
        return None
    row["feedback"] = extract_json(text)
    row["id"] = "gen-" + hashlib.sha1((row["question"] + row["answer"]).encode()).hexdigest()[:10]
    row["source"] = {"writer_model": args.writer_model, "teacher_model": args.teacher_model}
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="rows to attempt")
    p.add_argument("--out", default="data/generated/generated.jsonl")
    p.add_argument("--seed-file", default="data/seed/seed.jsonl")
    p.add_argument("--writer-model", default=os.environ.get("WRITER_MODEL", "openai/gpt-oss-120b"))
    p.add_argument("--teacher-model", default=os.environ.get("TEACHER_MODEL", "openai/gpt-oss-120b"))
    p.add_argument("--shots", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-mode", action="store_true", help="send response_format=json_object (if the model supports it)")
    p.add_argument("--require-calibrated", action="store_true", help="drop rows whose scores contradict the planned quality")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    seed = [json.loads(l) for l in open(args.seed_file, encoding="utf-8") if l.strip()]
    if not args.dry_run and not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY")):
        sys.exit("Set OPENROUTER_API_KEY (or LLM_API_KEY + LLM_BASE_URL). Use --dry-run to preview prompts.")
    if args.dry_run:
        for i in range(args.n):
            make_one(i, args, seed)
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = 0
    seen = set()
    with open(args.out, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(make_one, i, args, seed) for i in range(args.n)]
        for fut in cf.as_completed(futs):
            try:
                row = fut.result()
            except Exception as e:  # keep going on single failures
                print(f"row failed: {e}", file=sys.stderr)
                continue
            if row and row["id"] not in seen:
                seen.add(row["id"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    print(f"kept {kept}/{args.n} rows -> {args.out}")


if __name__ == "__main__":
    main()
