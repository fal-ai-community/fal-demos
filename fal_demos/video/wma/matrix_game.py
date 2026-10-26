"""Matrix-Game 2.0 on fal WMA: a keyboard-controlled world model over WebRTC.

The app loads Skywork's Matrix-Game 2.0 (MIT license), starts from a bundled
seed image, and streams generated video to the browser. The browser reports
held and tapped commands (move and look). The app turns them into the model's
keyboard and mouse conditioning for each generated chunk.
"""

import asyncio
import concurrent.futures
import logging
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

import fal
import fal.wma
from fal.wma.profile import CommandErrors, CommandState, CommandWorld, SerialAction

logger = logging.getLogger("matrix-game")

# Upstream code (MIT) and weights (MIT), both pinned.
UPSTREAM_URL = "https://github.com/SkyworkAI/Matrix-Game"
UPSTREAM_SHA = "c7a88bfc039cd97695082484986e95b962390676"
WEIGHTS_REPO = "Skywork/Matrix-Game-2.0"
WEIGHTS_SHA = "08c42d1a99e57e79df60a964bb815618cdf460f8"
DATA_DIR = Path("/data/matrix-game-2")

WIDTH, HEIGHT = 640, 352
FPS = 25  # Playback pace of the output video.
LATENT_PER_CHUNK = 3  # Upstream "num_frame_per_block".
FRAMES_PER_CHUNK = 12  # 3 latents decode to 12 frames (9 for the first chunk).
MAX_LATENT_FRAMES = 360  # Upstream default. About 1437 frames per world.
SCENE_COUNT = 17  # Images in demo_images/universal at the pinned commit.
CAMERA_STEP = 0.1  # Upstream mouse value for one camera step.

COMMANDS = (
    "forward",
    "back",
    "left",
    "right",
    "look_up",
    "look_down",
    "look_left",
    "look_right",
)
CONFLICTS = (
    ("forward", "back"),
    ("left", "right"),
    ("look_up", "look_down"),
    ("look_left", "look_right"),
)
Commands = fal.wma.commands_message(
    "Commands", commands=COMMANDS, conflict_groups=CONFLICTS
)


class Setup(BaseModel):
    """Choose the seed image. Accepted once, and restarts the world."""

    type: Literal["setup"] = "setup"
    scene: int = Field(default=0, ge=0, lt=SCENE_COUNT)


class Configured(BaseModel):
    type: Literal["configured"] = "configured"
    scene: int


class Reset(BaseModel):
    """Restart the world, optionally from another seed image."""

    type: Literal["reset"] = "reset"
    scene: Optional[int] = Field(default=None, ge=0, lt=SCENE_COUNT)


class ResetApplied(BaseModel):
    type: Literal["reset_applied"] = "reset_applied"
    scene: int


ClientMessage = Annotated[Union[Commands, Setup, Reset], Field(discriminator="type")]
ServerMessage = Annotated[
    Union[
        Configured, ResetApplied, fal.wma.ErrorMessage, fal.wma.StreamExhaustedMessage
    ],
    Field(discriminator="type"),
]

CONTRACT = fal.wma.RealtimeContract(
    media=fal.wma.MediaContract(
        receive=(fal.wma.Track(kind="video", width=WIDTH, height=HEIGHT),)
    ),
    client_messages=ClientMessage,
    server_messages=ServerMessage,
    interaction=CommandWorld(
        commands=CommandState(Commands, "/active", "/activated"),
        setup=SerialAction(Setup, Configured),
        reset=SerialAction(Reset, ResetApplied),
        errors=CommandErrors(fal.wma.ErrorMessage, "/error"),
        ended=fal.wma.StreamExhaustedMessage,
    ),
    client_message_examples=(
        fal.wma.MessageExample(
            name="Walk forward", payload={"type": "commands", "active": ["forward"]}
        ),
        fal.wma.MessageExample(
            name="Turn right while walking",
            payload={"type": "commands", "active": ["forward", "look_right"]},
        ),
        fal.wma.MessageExample(
            name="Stop", payload={"type": "commands", "active": []}
        ),
        fal.wma.MessageExample(name="Restart", payload={"type": "reset"}),
    ),
)

KEY_ORDER = tuple(name.upper() for name in COMMANDS)
KEY_CONFLICTS = tuple((a.upper(), b.upper()) for a, b in CONFLICTS)


def axis(keys: Dict[str, bool], positive: str, negative: str) -> float:
    """+1, -1 or 0. Opposite keys held together cancel."""
    return float(keys[positive]) - float(keys[negative])


def conditioning(keys: Dict[str, bool]):
    """Map sampled keys to Matrix-Game's [W, S, A, D] keyboard and [pitch, yaw] mouse."""
    forward, back = keys["FORWARD"], keys["BACK"]
    left, right = keys["LEFT"], keys["RIGHT"]
    if forward and back:
        forward = back = False
    if left and right:
        left = right = False
    keyboard = [float(forward), float(back), float(left), float(right)]
    mouse = [
        CAMERA_STEP * axis(keys, "LOOK_UP", "LOOK_DOWN"),
        CAMERA_STEP * axis(keys, "LOOK_RIGHT", "LOOK_LEFT"),
    ]
    return keyboard, mouse


@dataclass
class World:
    """Per-session generation state."""

    conditions: Dict[str, Any]
    noise: Any
    vae_cache: List[Any]
    start_latent: int = 0

    @property
    def finished(self) -> bool:
        return self.start_latent + LATENT_PER_CHUNK > MAX_LATENT_FRAMES


class Engine:
    """Loads the pinned upstream code and weights, and generates one chunk at a time."""

    def load(self) -> None:
        import sys

        import torch
        from huggingface_hub import snapshot_download

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        repo = DATA_DIR / f"Matrix-Game-{UPSTREAM_SHA}"
        if not (repo / "Matrix-Game-2").is_dir():
            # Clone into a private folder, then move it into place. A failed or
            # concurrent first start then never leaves a partial clone in /data.
            staging = Path(tempfile.mkdtemp(dir=DATA_DIR, prefix="clone-"))
            staging.chmod(0o755)  # mkdtemp uses 0700, and this folder becomes the clone.
            try:
                for args in (
                    ("init", "-q"),
                    ("remote", "add", "origin", UPSTREAM_URL),
                    ("sparse-checkout", "set", "Matrix-Game-2"),
                    ("fetch", "-q", "--depth", "1", "origin", UPSTREAM_SHA),
                    ("checkout", "-q", "FETCH_HEAD"),
                ):
                    subprocess.run(["git", "-C", str(staging), *args], check=True)
                try:
                    staging.rename(repo)
                except OSError:
                    # Another runner moved its clone into place first.
                    if not (repo / "Matrix-Game-2").is_dir():
                        raise
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        self.code = repo / "Matrix-Game-2"
        sys.path.insert(0, str(self.code))

        weights = Path(
            snapshot_download(
                WEIGHTS_REPO,
                revision=WEIGHTS_SHA,
                local_dir=DATA_DIR / "weights",
                allow_patterns=[
                    "Wan2.1_VAE.pth",
                    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                    "xlm-roberta-large/*",
                    "base_distilled_model/*",
                ],
            )
        )

        # Upstream modules. They resolve relative to the repo, so import after sys.path.
        from demo_utils.vae_block3 import VAEDecoderWrapper
        from omegaconf import OmegaConf
        from pipeline import CausalInferenceStreamingPipeline
        from safetensors.torch import load_file
        from torchvision.transforms import v2
        from utils.wan_wrapper import WanDiffusionWrapper
        from wan.vae.wanx_vae import get_wanx_vae_wrapper

        self.torch = torch
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        config = OmegaConf.load(
            self.code / "configs/inference_yaml/inference_universal.yaml"
        )

        generator = WanDiffusionWrapper(
            model_config=str(self.code / "configs/distilled_model/universal"),
            timestep_shift=config.model_kwargs.timestep_shift,
            is_causal=True,
        )
        decoder = VAEDecoderWrapper()
        vae_weights = torch.load(weights / "Wan2.1_VAE.pth", map_location="cpu")
        decoder.load_state_dict(
            {k: v for k, v in vae_weights.items() if "decoder." in k or "conv2" in k}
        )
        decoder.to(self.device, torch.float16).requires_grad_(False).eval()

        pipeline = CausalInferenceStreamingPipeline(
            config, generator=generator, vae_decoder=decoder
        )
        pipeline.generator.load_state_dict(
            load_file(str(weights / "base_distilled_model/base_distill.safetensors"))
        )
        self.pipeline = pipeline.to(device=self.device, dtype=self.dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        vae = get_wanx_vae_wrapper(str(weights), torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.dtype)

        self.transform = v2.Compose(
            [
                v2.Resize(size=(HEIGHT, WIDTH), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.scenes = sorted((self.code / "demo_images/universal").iterdir())
        if len(self.scenes) != SCENE_COUNT:
            raise RuntimeError("seed images differ from the pinned upstream commit")
        self.encoded: Dict[int, Dict[str, Any]] = {}

    def encode_scene(self, scene: int) -> Dict[str, Any]:
        """Encode a seed image into the image conditioning, once per scene."""
        if scene in self.encoded:
            return self.encoded[scene]
        from PIL import Image

        torch = self.torch
        image = Image.open(self.scenes[scene]).convert("RGB")
        # Center-crop to the model's 640x352 aspect ratio, as upstream does.
        w, h = image.size
        if h / w > HEIGHT / WIDTH:
            new_w, new_h = w, int(w * HEIGHT / WIDTH)
        else:
            new_w, new_h = int(h * WIDTH / HEIGHT), h
        left, top = (w - new_w) // 2, (h - new_h) // 2
        image = image.crop((left, top, left + new_w, top + new_h))
        tensor = self.transform(image)[None, :, None].to(self.device, self.dtype)

        padding = torch.zeros_like(tensor).repeat(
            1, 1, 4 * (MAX_LATENT_FRAMES - 1), 1, 1
        )
        encoded = self.vae.encode(
            torch.cat([tensor, padding], dim=2),
            device=self.device,
            tiled=True,
            tile_size=[44, 80],
            tile_stride=[23, 38],
        ).to(self.device)
        mask = torch.ones_like(encoded)
        mask[:, :, 1:] = 0
        self.encoded[scene] = {
            "cond_concat": torch.cat([mask[:, :4], encoded], dim=1).to(self.dtype),
            "visual_context": self.vae.clip.encode_video(tensor).to(self.dtype),
        }
        return self.encoded[scene]

    def new_world(self, scene: int) -> World:
        """Reset the KV caches and the VAE decoder cache, and start from a seed image."""
        from demo_utils.constant import ZERO_VAE_CACHE

        torch, pipe = self.torch, self.pipeline
        base = self.encode_scene(scene)
        frames = (MAX_LATENT_FRAMES - 1) * 4 + 1
        conditions = {
            **base,
            "keyboard_cond": torch.zeros(1, frames, 4, device=self.device).to(
                self.dtype
            ),
            "mouse_cond": torch.zeros(1, frames, 2, device=self.device).to(self.dtype),
        }
        pipe._initialize_kv_cache(1, self.dtype, self.device)
        pipe._initialize_kv_cache_mouse_and_keyboard(1, self.dtype, self.device)
        pipe._initialize_crossattn_cache(1, self.dtype, self.device)
        noise = torch.randn(
            [1, 16, MAX_LATENT_FRAMES, 44, 80], device=self.device, dtype=self.dtype
        )
        return World(conditions, noise, [None] * len(ZERO_VAE_CACHE))

    def step(self, world: World, keys: Dict[str, bool]):
        """Generate one chunk under the given keys. Returns uint8 frames [T, H, W, 3]."""
        from einops import rearrange
        from pipeline.causal_inference import cond_current

        torch, pipe = self.torch, self.pipeline
        keyboard, mouse = conditioning(keys)
        action = {
            "keyboard": torch.tensor(keyboard, device=self.device),
            "mouse": torch.tensor(mouse, device=self.device),
        }
        start = world.start_latent
        block = LATENT_PER_CHUNK
        step_cond, world.conditions = cond_current(
            world.conditions, start, block, replace=action, mode="universal"
        )
        noisy = world.noise[:, :, start : start + block]

        def call(latents, timestep):
            return pipe.generator(
                noisy_image_or_video=latents,
                conditional_dict=step_cond,
                timestep=timestep,
                kv_cache=pipe.kv_cache1,
                kv_cache_mouse=pipe.kv_cache_mouse,
                kv_cache_keyboard=pipe.kv_cache_keyboard,
                crossattn_cache=pipe.crossattn_cache,
                current_start=start * pipe.frame_seq_length,
            )[1]

        steps = pipe.denoising_step_list
        for index, current in enumerate(steps):
            timestep = torch.ones([1, block], device=self.device, dtype=torch.int64)
            timestep = timestep * current
            denoised = call(noisy, timestep)
            if index < len(steps) - 1:
                flat = rearrange(denoised, "b c f h w -> (b f) c h w")
                nxt = steps[index + 1] * torch.ones(
                    [block], device=self.device, dtype=torch.long
                )
                noisy = pipe.scheduler.add_noise(flat, torch.randn_like(flat), nxt)
                noisy = rearrange(noisy, "(b f) c h w -> b c f h w", b=1)
        # Write the clean chunk into the KV cache so the next chunk can attend to it.
        call(denoised, torch.ones_like(timestep) * pipe.args.context_noise)

        video, world.vae_cache = pipe.vae_decoder(
            denoised.transpose(1, 2).half(), *world.vae_cache
        )
        world.start_latent += block
        pixels = ((video[0].float() + 1) * 127.5).clamp(0, 255)
        return pixels.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()


class WorldSession:
    """One browser session: control state, a frame queue and a generation thread."""

    def __init__(self, engine: Engine, session: Any) -> None:
        self.engine = engine
        self.session = session
        self.loop = asyncio.get_running_loop()
        self.keys = fal.wma.KeyState(KEY_ORDER, KEY_CONFLICTS)
        # One chunk of frames. A full queue pauses generation, which limits latency.
        self.frames: "asyncio.Queue[Any]" = asyncio.Queue(maxsize=FRAMES_PER_CHUNK)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.restart: Optional[tuple] = None
        self.flushed = 0  # Billed frames dropped on restart. Event loop only.
        self.configured = False
        self.thread = threading.Thread(
            target=self.run, name="matrix-game", daemon=True
        )
        session.on_message("commands", self.on_commands)
        session.on_message("setup", self.on_setup)
        session.on_message("reset", self.on_reset)

    # Handlers run on the session event loop.
    def on_commands(self, message: Dict[str, Any]) -> None:
        try:
            parsed = Commands.model_validate(message)
        except ValueError as exc:
            self.session.send({"type": "error", "error": str(exc)[:200]})
            return
        self.keys.update(list(parsed.active), list(parsed.activated))

    def on_setup(self, message: Dict[str, Any]) -> None:
        if self.configured:
            self.session.send({"type": "error", "error": "setup is accepted once"})
            return
        try:
            parsed = Setup.model_validate(message)
        except ValueError as exc:
            self.session.send({"type": "error", "error": str(exc)[:200]})
            return
        self.configured = True
        self.request_restart(parsed.scene, Configured)

    def on_reset(self, message: Dict[str, Any]) -> None:
        try:
            parsed = Reset.model_validate(message)
        except ValueError as exc:
            self.session.send({"type": "error", "error": str(exc)[:200]})
            return
        self.request_restart(parsed.scene, ResetApplied)

    def request_restart(self, scene: Optional[int], ack: type) -> None:
        with self.lock:
            self.restart = (scene, ack)
        # Drop frames of the old world that wait for playback.
        while not self.frames.empty():
            self.frames.get_nowait()
            self.flushed += 1

    # Everything below runs on the generation thread.
    def take_restart(self) -> Optional[tuple]:
        with self.lock:
            restart, self.restart = self.restart, None
        return restart

    def push(self, frame: Any) -> bool:
        """Queue a frame. Block while the queue is full. False when the session stops."""
        future = asyncio.run_coroutine_threadsafe(self.frames.put(frame), self.loop)
        while not self.stop.is_set():
            try:
                future.result(timeout=0.2)
                return True
            except concurrent.futures.TimeoutError:
                continue
        future.cancel()
        return False

    def run(self) -> None:
        scene, ack = 0, None
        try:
            with self.engine.torch.no_grad():
                while not self.stop.is_set():
                    world = self.engine.new_world(scene)
                    if ack is not None:
                        self.session.send(ack(scene=scene).model_dump())
                    while not self.stop.is_set():
                        restart = self.take_restart()
                        if restart is not None:
                            scene = scene if restart[0] is None else restart[0]
                            ack = restart[1]
                            break
                        if world.finished:
                            self.finish()
                            return
                        frames = self.engine.step(world, self.keys.sample())
                        for frame in frames:
                            with self.lock:
                                if self.restart is not None:
                                    break
                            if not self.push(frame):
                                return
                            try:
                                self.session.add_billable_units(1)
                            except RuntimeError:
                                return  # Billing is final, so the session is closing.
        except Exception:
            logger.exception("generation failed")
            self.session.send({"type": "error", "error": "generation failed"})
            self.loop.call_soon_threadsafe(self.end_session)

    def finish(self) -> None:
        self.session.send({"type": "stream_exhausted"})
        self.loop.call_soon_threadsafe(self.end_session)

    def end_session(self) -> None:
        async def close_later() -> None:
            await asyncio.sleep(1)  # Let the last control message leave.
            await self.session.close()

        self.session.create_task(close_later())

    def cap_billing(self) -> None:
        """Stop billing for frames that were billed but never reached playback.

        Runs after the generation thread has stopped and before billing is finalized.
        """
        undelivered = self.flushed + self.frames.qsize()
        self.session.cap_billable_units(max(0.0, self.session.billable_units - undelivered))

    async def stop_and_join(self) -> None:
        self.stop.set()
        if self.thread.is_alive():
            await asyncio.to_thread(self.thread.join, 30)


class MatrixGameApp(
    fal.wma.App,
    python_version="3.12",
    keep_alive=300,
    min_concurrency=0,
    max_concurrency=2,
    max_multiplexing=1,
):
    machine_type = "GPU-H100"
    startup_timeout = 1800  # Loading about 12 GB of weights can take many minutes.
    requirements = [
        "fal[wma]==1.82.1",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
        "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
        "numpy==2.2.6",
        "pillow==12.3.0",
        "diffusers==0.41.0",
        "transformers==5.20.2",
        "tokenizers==0.23.3",
        "accelerate==1.15.0",
        "huggingface_hub==1.33.0",
        "omegaconf==2.4.0",
        "einops==0.8.2",
        "easydict==1.13",
        "ftfy==6.3.1",
        "regex==2026.9.29",
        "scipy==1.18.1",
        "safetensors==0.8.0",
        "sentencepiece==0.2.2",
        "open_clip_torch==3.3.0",
        "opencv-python-headless==5.0.0.93",
        "imageio==2.38.1",
        "av==17.1.0",
        "tqdm==4.70.1",
        "packaging==26.3",
        "requests==2.34.2",
        "pydantic>=2,<3",
    ]
    realtime_contract = CONTRACT

    def setup(self) -> None:
        self.engine = Engine()
        self.engine.load()
        # Warm the caches and decoder before the first session.
        world = self.engine.new_world(0)
        with self.engine.torch.no_grad():
            self.engine.step(world, {key: False for key in KEY_ORDER})
        self.slot = fal.wma.SessionSlot()

    async def create_backend(self, session):
        from aiortc import RTCConfiguration

        # One session at a time: the KV caches belong to the runner, not the session.
        if not self.slot.try_acquire():
            raise fal.wma.AppError(
                status_code=503,
                errors=[
                    fal.wma.Error(
                        loc=["body"],
                        msg="Another session is active on this runner. Retry.",
                        type="session_busy",
                        input=None,
                    )
                ],
                retryable=True,
                billing_units=0,
            )
        session.defer(self.slot.release)
        world = WorldSession(self.engine, session)
        # Deferred cleanups run in reverse: join the thread, cap billing, release the slot.
        session.defer(world.cap_billing)
        session.defer(world.stop_and_join)

        servers, _ = await fal.wma.RunnerIceConfig.from_bridge().build_ice_servers_async(
            session.offer.ice_servers, forwarded_status=session.offer.ice_status
        )
        track = fal.wma.make_video_queue_track(world.frames, FPS)
        world.thread.start()
        return fal.wma.AiortcPeer(
            session,
            lambda pc: pc.addTrack(track),
            rtc_configuration=RTCConfiguration(iceServers=servers),
            disconnected_grace_seconds=10,
        )
