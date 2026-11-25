#!/usr/bin/env python3
"""Turn seed-format rows into Halo datasets.

Writes:
  data/sft/train.jsonl, data/sft/test.jsonl   Halo SFT rows: {"prompt": [system, user, assistant]}
  data/grpo/train.jsonl                       Halo online-GRPO rows: {"prompt": [system, user], "answer": <reference JSON>}
  data/eval/test.jsonl                        held-out rows in seed format, for scripts/eval.py

The split is deterministic (hash of the row id), so reruns produce the same split.

  python scripts/build_datasets.py --inputs data/seed/seed.jsonl data/generated/*.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from coach.rubric import build_messages, schema_errors  # noqa: E402


def is_test(row_id: str, frac: float) -> bool:
    return int(hashlib.sha1(row_id.encode()).hexdigest(), 16) % 10000 < frac * 10000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", default=["data/seed/seed.jsonl"])
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--outdir", default="data")
    a = p.parse_args()

    rows, seen = [], set()
    for path in a.inputs:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in seen or schema_errors(r["feedback"]):
                continue
            seen.add(r["id"])
            rows.append(r)

    for sub in ("sft", "grpo", "eval"):
        os.makedirs(os.path.join(a.outdir, sub), exist_ok=True)
    out = {k: open(os.path.join(a.outdir, *k.split("/")), "w", encoding="utf-8")
           for k in ("sft/train.jsonl", "sft/test.jsonl", "grpo/train.jsonl", "eval/test.jsonl")}
    n_train = n_test = 0
    for r in rows:
        msgs = build_messages(r)
        ref = json.dumps(r["feedback"], ensure_ascii=False)
        if is_test(r["id"], a.test_frac):
            out["sft/test.jsonl"].write(json.dumps({"prompt": msgs + [{"role": "assistant", "content": ref}]}, ensure_ascii=False) + "\n")
            out["eval/test.jsonl"].write(json.dumps(r, ensure_ascii=False) + "\n")
            n_test += 1
        else:
            out["sft/train.jsonl"].write(json.dumps({"prompt": msgs + [{"role": "assistant", "content": ref}]}, ensure_ascii=False) + "\n")
            out["grpo/train.jsonl"].write(json.dumps({"prompt": msgs, "answer": ref}, ensure_ascii=False) + "\n")
            n_train += 1
    for f in out.values():
        f.close()
    print(f"{len(rows)} rows -> train {n_train}, test {n_test} (in {a.outdir}/sft, grpo, eval)")


if __name__ == "__main__":
    main()
