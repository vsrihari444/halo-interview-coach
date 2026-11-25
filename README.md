# Halo Interview Coach

A small open model, post-trained with [Halo](https://github.com/whitecircle/halo), that gives **structured, grounded feedback on interview answers**. It reads a question, a candidate's answer and (optionally) facts the candidate has confirmed about themselves, then returns one JSON object:

```json
{
  "scores": {"clarity": 4, "specificity": 2, "structure": 3, "concept_coverage": 2},
  "strengths": ["Relevant engineering accomplishment."],
  "improvements": ["Use the numbers you already have: 40 minutes down to 6 minutes is a strong result, so say it."],
  "missing_concepts": ["quantified result", "technical approach"],
  "rewrite_tip": "'At Acme Logistics in 2024 I cut report generation from 40 minutes to 6 minutes by [technique]...'"
}
```

**The hard rule: the coach never invents facts about the candidate.** A rewrite may reuse only what the candidate said or confirmed; anything else must be a `[bracketed placeholder]`. Generic LLM coaches routinely "improve" an answer by making up metrics, which is actively harmful in an interview. This project trains that behavior out and measures it automatically.

Built by [Srihari Reddy Vunadi](https://github.com/vsrihari444) for the [Halo Competition](https://whitecircle.com/halo/competition) (Dec 1–10, 2026). It grows out of my work on Applymat, an AI interview-preparation product whose coaching reviews use the same four dimensions and the same grounding rule.

> **Status: pre-competition prep.** The rubric, seed data, data generator, eval harness and Halo configs are here and tested offline. Training runs, results and the model link will be added during the competition window. No results are claimed yet.

## How it works

| Step | What | Tech |
|---|---|---|
| 1. Seed set | 50 hand-written examples (20 strong, 16 medium, 14 weak answers; 9 with confirmed facts) across engineering, data, product, sales and behavioral questions | `data/seed/seed.jsonl` |
| 2. Synthetic data | A writer model drafts question + answer at a planned quality; a teacher model writes reference feedback with the seed as few-shot. Rows that fail the schema or grounding checks are dropped | OpenRouter (any OpenAI-compatible API), `scripts/generate_data.py` |
| 3. SFT | LoRA SFT of `Qwen/Qwen3-4B-Instruct-2507` on one GPU | Halo `sft`, `configs/sft-qwen3-4b-interview-coach-lora.yaml` |
| 3b. Gemma port | The same recipe on `google/gemma-4-26B-A4B-it` (LoRA, expert parallel) | Halo `sft`, `configs/sft-gemma4-26b-a4b-interview-coach-lora.yaml` (draft) |
| 4. RL (stage 2) | Online GRPO with Halo's built-in **judge** reward (OpenRouter-hosted) scoring score accuracy, actionability, grounding and schema, plus a JSON format reward | Halo `grpo/online`, `configs/grpo-qwen3-4b-interview-coach-judge.yaml` |
| 5. Eval | Base vs SFT vs GRPO on a held-out split: JSON/schema validity, **grounded rate** (no invented numbers), calibration on strong/weak answers, score MAE vs reference, plus an LLM judge | `scripts/eval.py` |
| 6. Compute | Serverless GPU launcher for the Halo image | Modal, `modal_app/train_sft.py` |

## Quick start

```bash
# 0. Offline checks (stdlib only, Python 3.10+)
python -m unittest discover -s tests
python scripts/build_datasets.py --inputs data/seed/seed.jsonl
python scripts/eval.py --data data/eval/test.jsonl --reference-as-prediction   # pipeline self-test

# 1. Generate synthetic data (needs OPENROUTER_API_KEY; see .env.example)
python scripts/generate_data.py --n 3000 --out data/generated/v1.jsonl --require-calibrated
python scripts/build_datasets.py --inputs data/seed/seed.jsonl data/generated/v1.jsonl

# 2. Train with Halo (inside the Halo image; mount this repo at /workspace/coach)
docker run --gpus all -it -v $PWD:/workspace/coach public.ecr.aws/whitecircle/halo:hopper bash
halo launch sft /workspace/coach/configs/sft-qwen3-4b-interview-coach-lora.yaml --max_steps=5 --save_strategy=no --report_to=none  # smoke
halo launch sft /workspace/coach/configs/sft-qwen3-4b-interview-coach-lora.yaml
#    or on Modal:  modal run modal_app/train_sft.py --full

# 3. Evaluate (serve the checkpoint with vLLM, then)
python scripts/eval.py --data data/eval/test.jsonl --model coach --base-url http://localhost:8000/v1 \
    --judge-model openai/gpt-oss-120b --out results/sft.json
```

## Rerun instructions (for the organizers)

_To be finalized by Dec 10 with the exact Halo version, dataset snapshot (published to the Hugging Face Hub), seeds, GPU type and expected metrics._

## Repo layout

```
coach/rubric.py        system prompt, JSON schema, grounding + calibration checks (shared by data, eval and tests)
scripts/               generate_data.py, build_datasets.py, eval.py, _llm.py (stdlib OpenAI-compatible client)
configs/               Halo configs: LoRA SFT (Qwen3-4B, Gemma 4 26B-A4B) and online GRPO with an OpenRouter judge
data/seed/seed.jsonl   50 hand-written examples
modal_app/             Modal launcher for the Halo image
tests/                 unit tests for the rubric checks and seed set
```

## Data and attribution

- Base model: [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (Apache-2.0). The trained adapter follows the base model's license.
- Training framework: [whitecircle/halo](https://github.com/whitecircle/halo). Configs adapted from its `examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml` and `examples/grpo/online/` recipes.
- Seed data: written by hand for this repo. All people, companies and numbers in it are fictional. No real candidate or user data is used anywhere.
- Synthetic data: generated through OpenRouter; the generating models are recorded per row (`source`).

## License

Apache-2.0 (see `LICENSE`). Modifications to Halo itself, if any, fall under the Halo License.
