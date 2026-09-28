"""
Flux LoRA training launcher without `fal.distributed`.

Runs inside the fal container as an ordinary Python process, started by
`app.py`'s `setup()`. It spawns one rank per GPU over plain NCCL:

    app.py  --HTTP-->  rank 0 (uvicorn)  --broadcast-->  ranks 1..N-1

Rank 0 serves a small HTTP server on localhost and broadcasts each job to the
other ranks, which sit in `dist.broadcast_object_list` between requests. The
training code itself is `flux_lora/core.py`, shared verbatim with the
`fal.distributed` variant in `trainer/`.

Swap the spawn loop below for `torchrun`, `accelerate launch` or Lightning
Fabric and nothing else has to change -- that is the point of this variant.
"""

import argparse
import datetime
import multiprocessing
import os
import sys
import time
from threading import Thread
from typing import Any

DEFAULT_PORT = 8001


def run_rank(
    rank: int,
    world_size: int,
    model_path: str,
    port: int,
    ready_file: str,
    weights_ready_file: str | None,
) -> None:
    """Entrypoint for one rank. Heavy imports happen here, in every child."""
    # Importing torch/diffusers takes tens of seconds. Doing it in the children
    # rather than the parent means all ranks import in parallel, and that time
    # overlaps the weight download the app is doing concurrently.
    import torch
    import torch.distributed as dist
    import uvicorn
    from fastapi import FastAPI

    from fal_demos.distributed.training.flux_lora.core import FluxLoRATrainerCore

    if weights_ready_file:
        # The app downloads weights while we import. Wait for it to confirm
        # they are complete before any rank touches them.
        while not os.path.exists(weights_ready_file):
            time.sleep(0.2)

    # A long timeout matters here: the non-zero ranks block in
    # broadcast_object_list between requests, which can be hours on a warm app.
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(days=365),
    )
    torch.cuda.set_device(rank)

    class Rank(FluxLoRATrainerCore):
        """Supplies the five things core.py expects from a launcher."""

        def __init__(self) -> None:
            self.rank = rank
            self.world_size = world_size
            self.device = torch.device(f"cuda:{rank}")

        def rank_print(self, message: str, debug: bool = False) -> None:
            prefix = "[debug] " if debug else ""
            print(f"{prefix}[rank {self.rank}] {message}", flush=True)

        # emit_progress stays the no-op from core: this variant returns the
        # final result over HTTP rather than streaming intermediate steps.

    trainer = Rank()
    trainer.setup_model(model_path=model_path)

    if rank != 0:
        # Block until rank 0 hands us a job. `None` payload means shut down.
        while True:
            params: list[Any] = [None]
            dist.broadcast_object_list(params, src=0)
            if params[0] is None:
                break
            trainer.run_training(**params[0])
        return

    app = FastAPI(title="Flux LoRA Trainer")

    @app.post("/")
    def train(payload: dict) -> dict:
        # Hand the job to the other ranks, then join the training ourselves.
        # core.run_training broadcasts the tensors internally, so every rank
        # must enter it together.
        dist.broadcast_object_list([payload], src=0)
        return trainer.run_training(**payload)

    def _signal_ready() -> None:
        # Written last, so the app's readiness poll only unblocks once every
        # rank has loaded its model and this server can actually serve.
        dist.barrier()
        with open(ready_file, "w") as handle:
            handle.write("ready")

    Thread(target=_signal_ready, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/flux_weights")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ready-file", default="trainer_ready.txt")
    parser.add_argument("--weights-ready-file", default=None)
    args = parser.parse_args()

    import torch

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")

    if os.path.exists(args.ready_file):
        os.remove(args.ready_file)

    world_size = torch.cuda.device_count()

    # `spawn`, never `fork`. The parent process may already have initialized
    # CUDA, and forking after that leaves the children with an unusable
    # context. This is the single most common way a working multi-GPU trainer
    # breaks when it is moved into a serving process.
    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(
            target=run_rank,
            args=(
                rank,
                world_size,
                args.model_path,
                args.port,
                args.ready_file,
                args.weights_ready_file,
            ),
            daemon=True,
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    print(f"[launcher] spawned {world_size} ranks", flush=True)

    # Supervise like torch's mp.spawn(join=True): if any rank dies, take the
    # rest down and exit nonzero, so the app's readiness poll sees a crashed
    # subprocess instead of hanging until startup_timeout.
    while True:
        for rank, process in enumerate(processes):
            if process.exitcode not in (None, 0):
                print(
                    f"[launcher] rank {rank} exited with {process.exitcode}; "
                    "terminating all ranks",
                    flush=True,
                )
                for other in processes:
                    other.terminate()
                sys.exit(1)
        if all(process.exitcode == 0 for process in processes):
            return
        time.sleep(0.5)


if __name__ == "__main__":
    main()
