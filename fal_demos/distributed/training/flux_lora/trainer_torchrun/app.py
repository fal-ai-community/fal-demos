"""
Flux LoRA Training App - bring your own launcher.

Same training code as `trainer/app.py`, but the ranks are started by a plain
subprocess instead of `fal.distributed`. Use this shape when you already have a
working DDP trainer (torchrun, accelerate, Lightning Fabric) and would rather
keep your launcher than port it to `DistributedWorker`.

The app process owns the HTTP surface and forwards each request to rank 0 over
localhost. See the README for the tradeoff against the `fal.distributed`
variant.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar

import fal
import requests
from fal.toolkit import File, download_file
from pydantic import BaseModel, Field

TRAINER_PORT = 8001
READY_FILE = "trainer_ready.txt"
WEIGHTS_READY_FILE = "trainer_weights_ready.txt"
MODEL_PATH = "/data/flux_weights"
LAUNCHER = str(Path(__file__).resolve().parent / "launcher.py")


class TrainingRequest(BaseModel):
    training_data_url: File = Field(
        description="Preprocessed training data (.pt file) from the preprocessor app",
        media_type="application/octet-stream",
    )
    learning_rate: float = Field(default=5e-4, description="Base learning rate")
    b_up_factor: float = Field(
        default=3.0, description="Learning rate multiplier for lora_B parameters"
    )
    steps: int = Field(default=250, ge=100, le=10000, description="Training steps")
    batch_size: int = Field(default=4, description="Batch size per GPU")
    gradient_accumulation_steps: int = Field(default=1)
    seed: int = Field(default=42)


class TrainingResponse(BaseModel):
    checkpoint: File = Field(description="Trained LoRA weights")
    final_loss: float = Field(description="Final average training loss")
    num_steps: int = Field(description="Number of steps completed")


class FluxLoRATorchrunApp(
    fal.App,
    name="flux-trainer-torchrun-demo",
    keep_alive=3000,
    min_concurrency=1,
    max_concurrency=1,
):
    """Flux LoRA trainer whose ranks are launched by a plain subprocess."""

    machine_type = "GPU-H100"
    num_gpus = 8
    # setup() downloads FLUX weights and waits for every rank to load a model.
    startup_timeout = 3600

    requirements: ClassVar[list[str]] = [
        "torch==2.4.0",
        "diffusers==0.30.3",
        "transformers==4.46.0",
        "tokenizers==0.20.1",
        "sentencepiece",
        "peft==0.12.0",
        "safetensors==0.4.4",
        "accelerate==1.4.0",
        "huggingface_hub==0.26.5",
        "uvicorn",
        "fastapi",
        "requests",
    ]

    def setup(self) -> None:
        from huggingface_hub import snapshot_download

        for stale in (READY_FILE, WEIGHTS_READY_FILE):
            if os.path.exists(stale):
                os.remove(stale)

        env = {**os.environ}
        env["HF_HOME"] = "/data/models"
        # Single-node NCCL. On NCCL 2.20.5 + Hopper the first collective can
        # fail with CUDA error 401 while NVLS sets up its memory pools; ring
        # and tree collectives are a safe fallback inside one node.
        env["NCCL_NVLS_ENABLE"] = "0"

        # Start the ranks FIRST. Their torch/diffusers imports take tens of
        # seconds, and that time is free if it overlaps the download below
        # rather than queueing behind it. The ranks wait on WEIGHTS_READY_FILE
        # before they touch /data.
        print("[setup] starting rank launcher...")
        self.process = subprocess.Popen(
            [
                sys.executable,
                LAUNCHER,
                "--model-path", MODEL_PATH,
                "--port", str(TRAINER_PORT),
                "--ready-file", READY_FILE,
                "--weights-ready-file", WEIGHTS_READY_FILE,
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
        )

        print("[setup] downloading FLUX.1-dev weights...")
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev", local_dir=MODEL_PATH
        )
        with open(WEIGHTS_READY_FILE, "w") as handle:
            handle.write("ready")
        print("[setup] weights ready, waiting for ranks...")

        deadline = time.monotonic() + self.startup_timeout
        while not os.path.exists(READY_FILE):
            returncode = self.process.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"rank launcher exited with {returncode} before becoming ready"
                )
            if time.monotonic() > deadline:
                self.process.kill()
                raise RuntimeError("ranks did not become ready before startup_timeout")
            time.sleep(1.0)

        print("[setup] all ranks ready")

    @fal.endpoint("/train")
    def train(self, request: TrainingRequest) -> TrainingResponse:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = str(
                download_file(request.training_data_url.url, target_dir=temp_dir)
            )

            # No client-side timeout: training is the request.
            response = requests.post(
                f"http://127.0.0.1:{TRAINER_PORT}/",
                json={
                    "training_data_path": data_path,
                    "learning_rate": request.learning_rate,
                    "b_up_factor": request.b_up_factor,
                    "max_train_steps": request.steps,
                    "batch_size": request.batch_size,
                    "gradient_accumulation_steps": request.gradient_accumulation_steps,
                    "guidance_scale": 1.0,
                    "use_masks": False,
                    "lr_scheduler": "constant",
                    "lr_warmup_steps": 0,
                    "seed": request.seed,
                    "streaming": False,
                },
                timeout=None,
            )
            response.raise_for_status()
            result = response.json()

            return TrainingResponse(
                checkpoint=File.from_path(result["checkpoint_path"]),
                final_loss=result["final_loss"],
                num_steps=result["num_steps"],
            )
