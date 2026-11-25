"""Run the Halo SFT recipe on Modal (serverless GPU). DRAFT: validate on Dec 1 with competition credits.

  pip install modal && modal setup            # uses your own Modal account / competition credits
  modal run modal_app/train_sft.py            # smoke run (5 steps)
  modal run modal_app/train_sft.py --full     # full LoRA SFT

Checkpoints land in the Modal volume "halo-interview-coach" under /workspace/coach/checkpoints.
"""
import os
import subprocess

import modal

HALO_IMAGE = "public.ecr.aws/whitecircle/halo:hopper"   # H100/H200; use :blackwell on B200
CONFIG = "/workspace/coach/configs/sft-qwen3-4b-interview-coach-lora.yaml"

image = (
    modal.Image.from_registry(HALO_IMAGE)
    .add_local_dir("configs", "/workspace/coach/configs")
    .add_local_dir("data", "/workspace/coach/data")
)
volume = modal.Volume.from_name("halo-interview-coach", create_if_missing=True)
app = modal.App("halo-interview-coach", image=image)


@app.function(gpu="H100", timeout=6 * 60 * 60, volumes={"/workspace/coach/checkpoints": volume})
def train(full: bool = False):
    cmd = ["halo", "launch", "sft", CONFIG]
    if not full:
        cmd += ["--max_steps=5", "--save_strategy=no", "--report_to=none"]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **os.environ})
    volume.commit()


@app.local_entrypoint()
def main(full: bool = False):
    train.remote(full=full)
