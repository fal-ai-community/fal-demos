"""Restyle a live webcam stream with StreamDiffusionV2 over fal WMA.

The browser sends its camera track. The app crops each frame to a square, feeds
the frames to StreamDiffusionV2 (Wan2.1 1.3B, single GPU) in chunks of four,
and sends the restyled frames back as a video track. A `prompt` control message
changes the style while the stream runs.
"""

import asyncio
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

import fal.wma

logger = logging.getLogger(__name__)

# Upstream code, pinned to one commit (Apache-2.0).
REPO = "chenfengxu714/StreamDiffusionV2"
REPO_SHA = "6961a5cf2045d1dda05a04ef229698bdc04e873a"
# Weights, pinned to one revision each (both Apache-2.0).
WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
WAN_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
CKPT_REPO = "jerryfeng/StreamDiffusionV2"
CKPT_REVISION = "2373eb2b39278b3a1aa174964a724ee78ead96f0"
DATA_ROOT = Path("/data/streamdiffusion-v2")

SIZE = 512  # Output is SIZE x SIZE. The upstream online demo uses 512 x 512.
FPS = 16  # The upstream default output rate.
CHUNK_FRAMES = 4  # Input frames per denoising step (4 * num_frame_per_block).
FIRST_CHUNK_FRAMES = 1 + CHUNK_FRAMES  # A new stream starts with one extra frame.
MAX_BACKLOG_FRAMES = 2 * CHUNK_FRAMES  # Waiting input frames. Older ones are dropped.
OUTPUT_QUEUE_FRAMES = FPS  # Generated frames waiting to play. Older ones are dropped.
# How much noise replaces the camera frame before denoising. The upstream default is 0.8,
# which mostly keeps the camera look. 0.85 gives a clear restyle and keeps faces and motion.
# At 0.9 faces start to change, and at 0.95 identity is lost.
NOISE_SCALE = 0.85
DENOISING_STEPS = 2  # Upstream recommends 2 for real-time use.
MAX_SESSION_SECONDS = 300

DEFAULT_PROMPT = (
    "Cyberpunk-inspired figure, neon-lit hair highlights, glowing interface "
    "holograms floating around, vibrant neon color palette, cinematic sci-fi style"
)


class Prompt(BaseModel):
    type: Literal["prompt"] = "prompt"
    text: str = Field(
        min_length=1,
        max_length=400,
        description="Style prompt. The stream restarts to apply it.",
    )


class Status(BaseModel):
    type: Literal["status"] = "status"
    state: Literal["waiting_for_camera", "running", "restarting"]
    prompt: str
    output_frames: int = 0
    dropped_input_frames: int = 0


class Rejected(BaseModel):
    type: Literal["rejected"] = "rejected"
    reason: str


ServerMessage = Annotated[Union[Status, Rejected], Field(discriminator="type")]


def prepare_runtime():
    """Download pinned code and weights, load the model, and warm it up."""
    import torch
    from huggingface_hub import snapshot_download

    root = DATA_ROOT / f"StreamDiffusionV2-{REPO_SHA}"
    if not (root / "configs" / "wan_causal_dmd_v2v.yaml").is_file():
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        # Download and extract into a private folder, then move the code into place.
        # A failed or concurrent first start then never leaves a partial tree in /data.
        staging = Path(tempfile.mkdtemp(dir=DATA_ROOT, prefix="extract-"))
        try:
            archive = staging / "code.tar.gz"
            urllib.request.urlretrieve(
                f"https://github.com/{REPO}/archive/{REPO_SHA}.tar.gz", archive
            )
            with tarfile.open(archive) as tar:
                # The archive is a pinned commit. Python 3.11 runners may lack extract filters.
                tar.extractall(staging)
            try:
                (staging / f"StreamDiffusionV2-{REPO_SHA}").rename(root)
            except OSError:
                # Another runner moved its copy into place first.
                if not (root / "configs" / "wan_causal_dmd_v2v.yaml").is_file():
                    raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    # The upstream code looks for wan_models/ and ckpts/ under this root.
    os.environ["STREAMDIFFUSIONV2_ROOT"] = str(root)
    snapshot_download(
        WAN_REPO,
        revision=WAN_REVISION,
        local_dir=root / "wan_models" / "Wan2.1-T2V-1.3B",
        ignore_patterns=["assets/*", "examples/*"],
    )
    snapshot_download(
        CKPT_REPO,
        revision=CKPT_REVISION,
        local_dir=root / "ckpts",
        allow_patterns=["wan_causal_dmd_v2v/*"],
    )

    sys.path.insert(0, str(root))
    from streamv2v.inference import SingleGPUInferencePipeline
    from streamv2v.inference_common import merge_cli_config

    torch.set_grad_enabled(False)
    config = merge_cli_config(
        str(root / "configs" / "wan_causal_dmd_v2v.yaml"),
        {
            "config_path": str(root / "configs" / "wan_causal_dmd_v2v.yaml"),
            "checkpoint_folder": str(root / "ckpts" / "wan_causal_dmd_v2v"),
            "noise_scale": NOISE_SCALE,
            "height": SIZE,
            "width": SIZE,
            "fps": FPS,
            "step": DENOISING_STEPS,
            "seed": 0,
            "model_type": "T2V-1.3B",
            "profile": False,
            "use_taehv": False,
            "use_tensorrt": False,
            "fast": False,
            "t2v": False,
            "target_fps": None,
            "fixed_noise_scale": False,
            "num_frames": 81,
            "normalize_latents": False,
        },
    )
    manager = SingleGPUInferencePipeline(config, torch.device("cuda"))
    manager.load_model(str(root / "ckpts" / "wan_causal_dmd_v2v"))

    # Run a short stream so kernel compilation happens before the first session.
    def noise(frames):
        return (
            torch.rand(1, 3, frames, SIZE, SIZE, device="cuda") * 2 - 1
        ).to(torch.bfloat16)

    stream, _ = manager.start_stream_session(
        "warm up", noise(FIRST_CHUNK_FRAMES), NOISE_SCALE
    )
    for _ in range(len(config.denoising_step_list) + 1):
        manager.run_stream_batch(stream, noise(CHUNK_FRAMES))
    return manager


def square_rgb(frame):
    """Center-crop an av.VideoFrame to a square and resize it to SIZE."""
    import av
    import numpy as np

    pixels = frame.to_ndarray(format="rgb24")
    height, width, _ = pixels.shape
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    square = np.ascontiguousarray(pixels[top : top + side, left : left + side])
    return (
        av.VideoFrame.from_ndarray(square, format="rgb24")
        .reformat(width=SIZE, height=SIZE)
        .to_ndarray()
    )


class RestyleSession:
    """Per-session buffers and the thread that runs the model."""

    def __init__(self, app, session, loop):
        self.app = app
        self.session = session
        self.loop = loop
        self.prompt = DEFAULT_PROMPT  # The latest requested prompt.
        self.state = "waiting_for_camera"
        self.latest = None  # Newest camera frame (av.VideoFrame).
        self.frames = deque()  # Input frames at FPS, oldest first.
        self.out = asyncio.Queue(maxsize=OUTPUT_QUEUE_FRAMES)
        self.cond = threading.Condition()
        self.stopped = threading.Event()
        self.thread = None
        self.output_frames = 0
        self.dropped_input_frames = 0
        self.dropped_output_frames = 0  # Billed frames that never played. Loop only.

    # Event loop side: control messages and camera input.

    def on_prompt(self, message):
        try:
            prompt = Prompt.model_validate(message)
        except ValueError:
            self.session.send(
                Rejected(reason="text must be 1 to 400 characters").model_dump()
            )
            return
        with self.cond:
            self.prompt = prompt.text
            self.cond.notify_all()

    def start(self, track):
        self.session.create_task(self.read_camera(track))
        self.session.create_task(self.sample_camera())
        self.session.create_task(self.report_status())
        self.thread = threading.Thread(target=self.run, name="restyle", daemon=True)
        self.thread.start()

    async def read_camera(self, track):
        # Drain the camera track so its queue never grows. Keep only the newest frame.
        while True:
            try:
                self.latest = await track.recv()
            except Exception:
                break
        # The camera ended. Stop feeding frames so the model stops generating.
        self.latest = None
        with self.cond:
            self.frames.clear()
        self.set_state("waiting_for_camera")

    async def sample_camera(self):
        # Resample the camera to FPS. Drop the oldest frames when the model falls behind.
        interval = 1 / FPS
        tick = time.monotonic()
        while True:
            tick += interval
            now = time.monotonic()
            if tick < now - 1:
                tick = now
            await asyncio.sleep(max(0.0, tick - now))
            if self.latest is None:
                continue
            pixels = square_rgb(self.latest)
            with self.cond:
                self.frames.append(pixels)
                while len(self.frames) > MAX_BACKLOG_FRAMES:
                    self.frames.popleft()
                    self.dropped_input_frames += 1
                self.cond.notify_all()

    async def report_status(self):
        while True:
            await asyncio.sleep(2)
            self.send_status()

    def send_status(self):
        self.session.send(
            Status(
                state=self.state,
                prompt=self.prompt,
                output_frames=self.output_frames,
                dropped_input_frames=self.dropped_input_frames,
            ).model_dump()
        )

    def put_output(self, frame):
        """Queue a frame for playback on the event loop. Count the frame it displaces."""
        if self.out.full():
            self.dropped_output_frames += 1
        fal.wma.queue_put_drop_oldest(self.out, frame)

    async def cap_billing(self):
        """Stop billing for frames that were billed but never reached playback.

        Runs after the model thread has stopped and before billing is finalized.
        """
        await asyncio.sleep(0)  # Let queued put_output callbacks run first.
        undelivered = self.dropped_output_frames + self.out.qsize()
        self.session.cap_billable_units(max(0.0, self.session.billable_units - undelivered))

    async def shutdown(self):
        self.stopped.set()
        with self.cond:
            self.cond.notify_all()
        if self.thread is not None:
            await asyncio.to_thread(self.thread.join, 30)

    # Model thread.

    def wait_for(self, count, active, newest=False):
        """Return `count` input frames, [] if the prompt changed, None if stopped."""
        with self.cond:
            while len(self.frames) < count:
                if self.stopped.is_set():
                    return None
                if self.prompt != active:
                    return []
                self.cond.wait(timeout=0.5)
            if self.stopped.is_set():
                return None
            while newest and len(self.frames) > count:
                self.frames.popleft()
                self.dropped_input_frames += 1
            return [self.frames.popleft() for _ in range(count)]

    def to_tensor(self, frames):
        import numpy as np
        import torch

        batch = torch.from_numpy(np.stack(frames)).to("cuda")
        batch = batch.float() / 127.5 - 1.0  # [-1, 1], as upstream expects
        return batch.permute(3, 0, 1, 2).unsqueeze(0).to(torch.bfloat16)

    def emit(self, video):
        """Queue decoded frames (float32, T x H x W x 3, range 0 to 1) for playback."""
        import numpy as np

        pixels = (np.clip(video, 0, 1) * 255).astype(np.uint8)
        try:
            for frame in pixels:
                self.loop.call_soon_threadsafe(self.put_output, frame)
        except RuntimeError:  # The session loop is already closed.
            self.stopped.set()
            return
        self.output_frames += len(pixels)
        try:
            self.session.add_billable_units(len(pixels))
        except RuntimeError:  # Billing is already finalized.
            self.stopped.set()

    def set_state(self, state):
        self.state = state
        self.send_status()

    def run(self):
        import torch

        torch.set_grad_enabled(False)
        try:
            with self.app.gpu_lock:
                self.generate()
        except Exception:
            logger.exception("restyle generation failed")
            self.session.send(Rejected(reason="generation failed").model_dump())
            try:
                self.loop.call_soon_threadsafe(self.end_session)
            except RuntimeError:
                pass

    def end_session(self):
        async def close_later():
            await asyncio.sleep(1)  # Let the last control message leave.
            await self.session.close()

        self.session.create_task(close_later())

    def generate(self):
        manager = self.app.manager
        active = None
        stream = None
        while not self.stopped.is_set():
            if stream is None or self.prompt != active:
                # Upstream encodes the prompt when a stream starts. A new prompt
                # therefore starts a new stream from the newest camera frames.
                with self.cond:
                    active = self.prompt
                if stream is not None:
                    self.set_state("restarting")
                frames = self.wait_for(FIRST_CHUNK_FRAMES, active, newest=True)
                if frames is None:
                    return
                if not frames:
                    continue
                stream, first = manager.start_stream_session(
                    active, self.to_tensor(frames), NOISE_SCALE
                )
                self.emit(first)
                self.set_state("running")
                continue
            frames = self.wait_for(CHUNK_FRAMES, active)
            if frames is None:
                return
            if not frames:
                continue
            # Returns no frames while the denoising pipeline fills, then one
            # decoded chunk of four frames per call.
            for video in manager.run_stream_batch(stream, self.to_tensor(frames)):
                self.emit(video)


class RestyleApp(fal.wma.App, python_version="3.11"):
    machine_type = "GPU-H100"
    requirements = [
        "fal[wma]==1.82.1",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "diffusers==0.35.1",
        "transformers==4.54.0",
        "accelerate==1.10.1",
        "huggingface_hub>=0.34,<1",
        "einops==0.8.1",
        "ftfy==6.3.1",
        "omegaconf==2.3.0",
        "regex==2024.11.6",
        "sentencepiece==0.2.0",
        "av==13.1.0",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "numpy==1.24.4",
        "Pillow>=10.2.0,<13",
        "scikit-image==0.25.2",
        "pydantic>=2,<3",
        "aiortc",
    ]
    startup_timeout = 1800  # Weights download and kernel compilation take minutes.
    min_concurrency = 0
    max_concurrency = 2
    max_multiplexing = 1  # One session owns the GPU and the model state.
    keep_alive = 300
    realtime_contract = fal.wma.RealtimeContract(
        media=fal.wma.MediaContract(
            send=(
                fal.wma.Track(
                    kind="video",
                    source="camera",
                    required=True,
                    width=fal.wma.Constraint(ideal=640),
                    height=fal.wma.Constraint(ideal=480),
                    frame_rate=fal.wma.Constraint(ideal=30),
                ),
            ),
            receive=(
                fal.wma.Track(kind="video", width=SIZE, height=SIZE, frame_rate=FPS),
            ),
        ),
        client_messages=Prompt,
        server_messages=ServerMessage,
        client_message_examples=(
            fal.wma.MessageExample(
                name="Cyberpunk",
                payload={"type": "prompt", "text": DEFAULT_PROMPT},
            ),
            fal.wma.MessageExample(
                name="Watercolor",
                payload={
                    "type": "prompt",
                    "text": "A soft watercolor painting, loose brush strokes, pastel colors",
                },
            ),
            fal.wma.MessageExample(
                name="Claymation",
                payload={
                    "type": "prompt",
                    "text": "A claymation character, studio lighting, stop-motion style",
                },
            ),
        ),
    )

    async def setup(self):
        self.gpu_lock = threading.Lock()
        self.manager = await asyncio.to_thread(prepare_runtime)

    async def create_backend(self, session):
        from aiortc import RTCConfiguration

        (
            servers,
            _,
        ) = await fal.wma.RunnerIceConfig.from_bridge().build_ice_servers_async(
            session.offer.ice_servers, forwarded_status=session.offer.ice_status
        )
        restyle = RestyleSession(self, session, asyncio.get_running_loop())
        session.on_message("prompt", restyle.on_prompt)
        # Deferred cleanups run in reverse: stop the model thread, then cap billing.
        session.defer(restyle.cap_billing)
        session.defer(restyle.shutdown)

        def connect(pc):
            attached = False

            @pc.on("track")
            def incoming(track):
                nonlocal attached
                if track.kind == "video" and not attached:
                    attached = True
                    pc.addTrack(
                        fal.wma.make_video_queue_track(
                            restyle.out, fps=FPS, rebase_after_stall=True
                        )
                    )
                    restyle.start(track)

        async def expire():
            await asyncio.sleep(MAX_SESSION_SECONDS)
            await session.close()

        session.create_task(expire())
        return fal.wma.AiortcPeer(
            session,
            connect,
            rtc_configuration=RTCConfiguration(iceServers=servers),
            disconnected_grace_seconds=10,
        )
