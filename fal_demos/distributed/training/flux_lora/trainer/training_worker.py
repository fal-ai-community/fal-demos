"""
Flux LoRA training worker, `fal.distributed` launcher.

All of the training lives in `flux_lora/core.py`. This file only adapts it to
`fal.distributed`, which already supplies `rank`, `world_size`, `device` and
`rank_print`, so the adapter is three methods.

See `trainer_torchrun/` for the same core driven by a plain spawn instead.
"""

from typing import Any

from fal.distributed import DistributedWorker

from fal_demos.distributed.training.flux_lora.core import FluxLoRATrainerCore


class FluxLoRATrainingWorker(FluxLoRATrainerCore, DistributedWorker):
    """Flux LoRA training worker driven by `DistributedRunner`."""

    def setup(self, model_path: str = "/data/flux_weights", **kwargs: Any) -> None:
        self.setup_model(model_path=model_path, **kwargs)

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        return self.run_training(**kwargs)

    def emit_progress(self, payload: dict[str, Any]) -> None:
        # Surfaces on the app's /stream endpoint as Server-Sent Events.
        self.add_streaming_result(payload, as_text_event=True)
