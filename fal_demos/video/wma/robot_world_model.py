"""Chunked robot policy and world model: NVIDIA Cosmos3-Edge-Policy-DROID over WMA.

The client sets an instruction and an observation image. Each `step` message makes the
model predict one action chunk (32 steps x 8 values, DROID joint space) and the video
that the robot is expected to see while it runs that chunk.
"""

import asyncio
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError

import fal
import fal.wma
from fal.container import ContainerImage
from fal.wma import profile

COSMOS_FRAMEWORK_URL = "https://github.com/NVIDIA/cosmos-framework.git"
COSMOS_FRAMEWORK_SHA = "9a70e64d2ad7ff1417484d12d10fb121b270baaf"
# Weights, pinned to one revision (NVIDIA OpenMDW 1.1, which allows commercial use).
CHECKPOINT = "nvidia/Cosmos3-Edge-Policy-DROID"
CHECKPOINT_REVISION = "a7c7288f9b6ac1684e993007b0f9703dd26e58ef"

# From the checkpoint's policy config: 32 action steps per chunk, conditioned at 15 fps.
FPS = 15
CHUNK_STEPS = 32
ACTION_DIM = 8
FRAME_QUEUE_SIZE = 4 * (CHUNK_STEPS + 1)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
SESSION_SECONDS = 900

DOCKERFILE = f"""
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \\
    ca-certificates curl ffmpeg git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/0.11.29/install.sh \\
    | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN git clone {COSMOS_FRAMEWORK_URL} /opt/cosmos-framework \\
    && git -C /opt/cosmos-framework checkout {COSMOS_FRAMEWORK_SHA}
WORKDIR /opt/cosmos-framework
RUN uv sync --locked --no-install-project --no-editable --all-extras \\
    --group=cu130-torch213-train \\
    && uv pip install --no-deps . \\
    && chmod -R a+rX /opt/uv-python /opt/cosmos-framework
ENV PATH=/opt/cosmos-framework/.venv/bin:$PATH \\
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
RUN uv pip install 'fal[wma]==1.82.1'
"""

# Franka-style ready pose in radians. Replace it with your robot's real state.
HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


class ConfigureMessage(BaseModel):
    type: Literal["configure"] = "configure"
    instruction_version: int = Field(ge=1, le=2**31)
    instruction: str = Field(min_length=1, max_length=2000)
    image_url: str = Field(
        max_length=4096,
        description=(
            "DROID-style observation: wrist camera on top, two exterior cameras "
            "side by side below."
        ),
    )
    joint_position: List[float] = Field(default=HOME_JOINTS, min_length=7, max_length=7)
    gripper_position: float = Field(default=0.0, ge=0, le=1)


class InstructionMessage(BaseModel):
    type: Literal["instruction"] = "instruction"
    instruction_version: int = Field(ge=1, le=2**31)
    instruction: str = Field(min_length=1, max_length=2000)


class StepMessage(BaseModel):
    type: Literal["step"] = "step"
    step_id: int = Field(ge=1, le=2**31)
    joint_position: Optional[List[float]] = Field(
        default=None,
        min_length=7,
        max_length=7,
        description="Measured joints. Omit to continue from the last predicted action.",
    )
    gripper_position: Optional[float] = Field(default=None, ge=0, le=1)
    image_url: Optional[str] = Field(
        default=None,
        max_length=4096,
        description="New observation. Omit to continue from the last predicted frame.",
    )


class ConfiguredMessage(BaseModel):
    type: Literal["configured"] = "configured"
    instruction_version: int
    chunk_steps: int = CHUNK_STEPS
    action_dim: int = ACTION_DIM
    fps: int = FPS


class InstructionPendingMessage(BaseModel):
    type: Literal["instruction_pending"] = "instruction_pending"
    instruction_version: int


class InstructionAppliedMessage(BaseModel):
    type: Literal["instruction_applied"] = "instruction_applied"
    instruction_version: int


class InstructionRejectedMessage(BaseModel):
    type: Literal["instruction_rejected"] = "instruction_rejected"
    instruction_version: int
    reason: str


class ActionChunkMessage(BaseModel):
    type: Literal["action_chunk"] = "action_chunk"
    step_id: int
    instruction_version: int
    chunk_index: int
    actions: List[List[float]] = Field(description="CHUNK_STEPS rows of 7 joints + gripper.")
    fps: int = FPS
    frames: int = Field(description="Predicted video frames decoded for this chunk.")
    generation_seconds: float


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    code: Literal[
        "invalid_configuration",
        "invalid_image",
        "generation_failed",
        "invalid_message",
        "not_configured",
        "already_configured",
        "busy",
    ]
    error: str
    instruction_version: Optional[int] = None
    step_id: Optional[int] = None


ClientMessage = Annotated[
    Union[ConfigureMessage, InstructionMessage, StepMessage],
    Field(discriminator="type"),
]
ServerMessage = Annotated[
    Union[
        ConfiguredMessage,
        InstructionPendingMessage,
        InstructionAppliedMessage,
        InstructionRejectedMessage,
        ActionChunkMessage,
        ErrorMessage,
    ],
    Field(discriminator="type"),
]


def fetch_image(url: str):
    """Download with fal's SSRF-safe client and return an RGB uint8 array."""
    import numpy as np
    from fal.toolkit.utils.ssrf import ssrf_safe_get
    from PIL import Image

    data = ssrf_safe_get(url, max_size=MAX_IMAGE_BYTES).content
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


class RobotWorldModel(fal.wma.App):
    machine_type = "GPU-H100"
    image = ContainerImage.from_dockerfile_str(DOCKERFILE)
    min_concurrency = 0
    max_concurrency = 1
    max_multiplexing = 1
    keep_alive = 300
    startup_timeout = 1800

    realtime_contract = fal.wma.RealtimeContract(
        media=fal.wma.MediaContract(receive=(fal.wma.Track(kind="video"),)),
        client_messages=ClientMessage,
        server_messages=ServerMessage,
        client_message_examples=(
            fal.wma.MessageExample(
                name="Configure",
                payload={
                    "type": "configure",
                    "instruction_version": 1,
                    "instruction": "Put the marker in the cup.",
                    "image_url": "https://v3b.fal.media/files/b/0aaf5cf4/vFh0QpYtc-Fqt-21peWW-_droid_observation.png",
                },
            ),
            fal.wma.MessageExample(
                name="Step from predicted state",
                payload={"type": "step", "step_id": 1},
            ),
            fal.wma.MessageExample(
                name="New instruction",
                payload={
                    "type": "instruction",
                    "instruction_version": 2,
                    "instruction": "Open the drawer.",
                },
            ),
        ),
        interaction=profile.ConfiguredSession(
            setup=ConfigureMessage,
            ready=profile.Event(ConfiguredMessage, correlation="/instruction_version"),
            sequence=profile.Sequence(field="/instruction_version"),
            updates=profile.VersionedInput(
                message=InstructionMessage,
                input_field="/instruction",
                policy="replace-pending",
                pending=profile.Event(
                    InstructionPendingMessage, correlation="/instruction_version"
                ),
                applied=profile.Event(
                    InstructionAppliedMessage, correlation="/instruction_version"
                ),
                rejected=profile.Event(
                    InstructionRejectedMessage, correlation="/instruction_version"
                ),
            ),
            errors=profile.ErrorMapping(
                message=ErrorMessage,
                code_field="/code",
                description_field="/error",
                correlation="/instruction_version",
                session_failure=(
                    "invalid_configuration",
                    "invalid_image",
                    "generation_failed",
                ),
                diagnostic=(
                    "invalid_message",
                    "not_configured",
                    "already_configured",
                    "busy",
                ),
            ),
        ),
    )

    def setup(self):
        # Keep the weights on the persistent volume between runners.
        os.environ.setdefault("HF_HOME", "/data/hf")
        from cosmos_framework.scripts.action_policy_server_robolab import (
            RobolabPolicyService,
            RobolabServerArgs,
        )

        # The same service the upstream RoboLab websocket server wraps. We call
        # infer() directly instead of opening its websocket.
        self.service = RobolabPolicyService(
            RobolabServerArgs(
                checkpoint_path=CHECKPOINT,
                hf_revision=CHECKPOINT_REVISION,
                decode_video=True,
                format_prompt_as_json=True,
                guidance_interval=(960.0, 1001.0),
            )
        )
        self.service.infer(self.observation("warm up", blank_image(), HOME_JOINTS, 0.0))
        # One worker for all sessions: a chunk still running for a closed session
        # cannot overlap the next session's chunk.
        self.executor = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def observation(prompt, image, joints, gripper):
        return {
            "prompt": prompt,
            "observation/image": image,
            "observation/joint_position": joints,
            "observation/gripper_position": gripper,
        }

    async def create_backend(self, session):
        from aiortc import RTCConfiguration

        (
            servers,
            _,
        ) = await fal.wma.RunnerIceConfig.from_bridge().build_ice_servers_async(
            session.offer.ice_servers, forwarded_status=session.offer.ice_status
        )
        frames = asyncio.Queue(maxsize=FRAME_QUEUE_SIZE)
        robot = RobotSession(
            session, self.service, self.executor, self.observation, frames
        )
        session.on_message("configure", robot.on_configure)
        session.on_message("instruction", robot.on_instruction)
        session.on_message("step", robot.on_step)
        session.defer(robot.shutdown)

        async def expire():
            await asyncio.sleep(SESSION_SECONDS)
            await session.close()

        session.create_task(expire())
        track = fal.wma.make_video_queue_track(frames, FPS)
        return fal.wma.AiortcPeer(
            session,
            lambda pc: pc.addTrack(track),
            rtc_configuration=RTCConfiguration(iceServers=servers),
            disconnected_grace_seconds=10,
        )


def blank_image(height=540, width=640):
    import numpy as np

    return np.zeros((height, width, 3), dtype=np.uint8)


class RobotSession:
    """Per-session state. Model calls run on one worker thread, never on the loop."""

    def __init__(self, session, service, executor, build_observation, frames):
        self.session = session
        self.service = service
        self.executor = executor
        self.build_observation = build_observation
        self.frames = frames
        self.configured = False
        self.configuring = False
        self.busy = False
        self.closed = False
        self.instruction = ""
        self.version = 0
        self.pending = None
        self.image = None
        self.joints = None
        self.gripper = 0.0
        self.chunk_index = 0

    def shutdown(self):
        # A chunk already running cannot be interrupted. Its result is dropped.
        self.closed = True

    def send(self, message):
        self.session.send(message.model_dump())

    def fail(self, code, text, **fields):
        self.send(ErrorMessage(code=code, error=text, **fields))

    async def end_with(self, code, text, **fields):
        self.fail(code, text, **fields)
        await asyncio.sleep(1)  # Let the last control message leave.
        await self.session.close()

    def invalid(self, message, exc):
        self.fail("invalid_message", str(exc.errors()[0]["msg"]))

    async def on_configure(self, raw):
        if self.configured or self.configuring:
            return self.fail("already_configured", "Send configure once per session.")
        try:
            message = ConfigureMessage.model_validate(raw)
        except ValidationError as exc:
            return await self.end_with("invalid_configuration", str(exc.errors()[0]["msg"]))
        self.configuring = True
        try:
            image = await asyncio.to_thread(fetch_image, message.image_url)
        except Exception as exc:
            return await self.end_with(
                "invalid_image", f"Could not load the image: {type(exc).__name__}"
            )
        self.image = image
        self.instruction = message.instruction
        self.version = message.instruction_version
        self.joints = message.joint_position
        self.gripper = message.gripper_position
        self.configured = True
        self.send(ConfiguredMessage(instruction_version=self.version))

    def on_instruction(self, raw):
        if not self.configured:
            return self.fail("not_configured", "Send configure first.")
        try:
            message = InstructionMessage.model_validate(raw)
        except ValidationError as exc:
            version = raw.get("instruction_version")
            if isinstance(version, int):
                return self.send(
                    InstructionRejectedMessage(
                        instruction_version=version, reason="invalid_instruction"
                    )
                )
            return self.invalid(raw, exc)
        newest = max(self.version, self.pending[0] if self.pending else 0)
        if message.instruction_version <= newest:
            return self.send(
                InstructionRejectedMessage(
                    instruction_version=message.instruction_version,
                    reason="stale_instruction_version",
                )
            )
        # The newest text wins. It reaches the model at the start of the next chunk.
        self.pending = (message.instruction_version, message.instruction)
        self.send(InstructionPendingMessage(instruction_version=message.instruction_version))

    async def on_step(self, raw):
        if not self.configured:
            return self.fail("not_configured", "Send configure first.")
        try:
            message = StepMessage.model_validate(raw)
        except ValidationError as exc:
            return self.invalid(raw, exc)
        if self.busy:
            return self.fail("busy", "A chunk is still generating.", step_id=message.step_id)
        self.busy = True
        try:
            await self.run_chunk(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.closed:
                await self.end_with(
                    "generation_failed", type(exc).__name__, step_id=message.step_id
                )
        finally:
            self.busy = False

    async def run_chunk(self, step):
        if step.image_url is not None:
            try:
                self.image = await asyncio.to_thread(fetch_image, step.image_url)
            except Exception as exc:
                if not self.closed:
                    await self.end_with(
                        "invalid_image",
                        f"Could not load the image: {type(exc).__name__}",
                        step_id=step.step_id,
                    )
                return
        if self.pending is not None:
            self.version, self.instruction = self.pending
            self.pending = None
            self.send(InstructionAppliedMessage(instruction_version=self.version))
        joints = step.joint_position if step.joint_position is not None else self.joints
        gripper = step.gripper_position if step.gripper_position is not None else self.gripper
        observation = self.build_observation(self.instruction, self.image, joints, gripper)

        started = time.monotonic()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, self.service.infer, observation)
        seconds = time.monotonic() - started
        if self.closed:
            return

        actions = result["action"]
        video = result["video"]
        # Frame 0 is assumed to repeat the conditioning observation. Show it only once.
        for frame in video[(0 if self.chunk_index == 0 else 1) :]:
            fal.wma.queue_put_drop_oldest(self.frames, frame.copy())
        # Without client input, the next chunk starts from this chunk's end.
        self.joints = [float(v) for v in actions[-1][:7]]
        self.gripper = min(max(float(actions[-1][7]), 0.0), 1.0)
        self.image = video[-1].copy()
        self.send(
            ActionChunkMessage(
                step_id=step.step_id,
                instruction_version=self.version,
                chunk_index=self.chunk_index,
                actions=actions.astype(float).tolist(),
                frames=len(video),
                generation_seconds=round(seconds, 3),
            )
        )
        self.chunk_index += 1
        # One billable unit per completed chunk.
        self.session.add_billable_units(1)
