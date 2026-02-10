from contextlib import suppress
from typing import Annotated, AsyncIterator, Literal

import fal
from fal.toolkit import clone_repository
from pydantic import BaseModel, Field, RootModel, TypeAdapter, ValidationError


class IceCandidate(BaseModel):
    candidate: str
    sdpMid: str | None = None
    sdpMLineIndex: int | None = None


class OfferInput(BaseModel):
    type: Literal["offer"]
    sdp: str


class IceCandidateInput(BaseModel):
    type: Literal["icecandidate"]
    candidate: IceCandidate | None = None


class ActionInput(BaseModel):
    type: Literal["action"]
    action: str


class PauseInput(BaseModel):
    type: Literal["pause"]
    paused: bool = False


RealtimeInputMessage = Annotated[
    OfferInput | IceCandidateInput | ActionInput | PauseInput,
    Field(discriminator="type"),
]


class RealtimeInput(RootModel):
    root: RealtimeInputMessage


class IceServersOutput(BaseModel):
    type: Literal["iceservers"]
    iceservers: list[dict]


class AnswerOutput(BaseModel):
    type: Literal["answer"]
    sdp: str


class IceCandidateOutput(BaseModel):
    type: Literal["icecandidate"]
    candidate: IceCandidate | None = None


class PauseOutput(BaseModel):
    type: Literal["pause"]
    paused: bool


class StreamExhaustedOutput(BaseModel):
    type: Literal["stream_exhausted"]


class ErrorOutput(BaseModel):
    type: Literal["error"]
    error: str


RealtimeOutputMessage = Annotated[
    IceServersOutput
    | AnswerOutput
    | IceCandidateOutput
    | PauseOutput
    | StreamExhaustedOutput
    | ErrorOutput,
    Field(discriminator="type"),
]


class RealtimeOutput(RootModel):
    root: RealtimeOutputMessage


class MatrixWebRTC2(fal.App):
    machine_type = "GPU-H100"
    startup_timeout = 1200
    TURN_EXPIRY_SECONDS = 600

    requirements = [
        "accelerate>=1.1.1",
        "aiortc",
        "av",
        "dashscope",
        "diffusers",
        "dominate",
        "easydict",
        "einops",
        "flask",
        "flask-socketio",
        "ftfy",
        "git+https://github.com/openai/CLIP.git",
        "huggingface-hub[cli]",
        "imageio",
        "imageio-ffmpeg",
        "lmdb",
        "matplotlib",
        "nvidia-tensorrt",
        "numpy",
        "omegaconf",
        "onnx",
        "onnxconverter_common",
        "onnxruntime",
        "onnxscript",
        "open_clip_torch",
        "opencv-python>=4.9.0.80",
        "pydantic",
        "pycocotools",
        "safetensors",
        "scikit-image",
        "sentencepiece",
        "starlette",
        "tokenizers>=0.20.3",
        "torch==2.6.0",
        "torchao==0.12.0",
        "torchvision",
        "tqdm",
        "transformers>=4.49.0",
        "wandb",
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    ]

    def setup(self):
        import os
        import re
        import subprocess
        import sys
        import threading

        self._repo_path = clone_repository(
            "https://github.com/efiop/Matrix-Game.git",
            commit_hash="50af8cb6e801caae43dae96f10d34626f38dd2e1",
        )
        self._matrix_repo_dir = self._repo_path / "Matrix-Game-2"
        if not self._matrix_repo_dir.exists():
            raise RuntimeError(
                f"Expected Matrix-Game-2 in cloned repo, not found at {self._matrix_repo_dir}"
            )

        sys.path.insert(0, str(self._matrix_repo_dir))
        os.chdir(self._matrix_repo_dir)

        cache_version = os.getenv("MATRIX_CACHE_VERSION", "v1")
        gpu_slug = "unknown"
        try:
            import torch

            gpu_name = torch.cuda.get_device_name(0)
            gpu_slug = (
                re.sub(r"[^a-zA-Z0-9]+", "-", gpu_name.strip()).strip("-").lower()
            )
        except Exception:
            pass

        inductor_cache_dir = (
            f"/data/inductor-cache/matrix-game-2/{gpu_slug}/{cache_version}"
        )
        triton_cache_dir = (
            f"/data/triton-cache/matrix-game-2/{gpu_slug}/{cache_version}"
        )
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", inductor_cache_dir)
        os.environ.setdefault("TRITON_CACHE_DIR", triton_cache_dir)

        subprocess.run(
            [
                "hf",
                "download",
                "Skywork/Matrix-Game-2.0",
                "--local-dir",
                "/data/Matrix-Game-2.0",
            ],
            check=True,
        )

        self._metered_secret_key = os.getenv("METERED_TURN_SECRET_KEY")
        self._metered_label = os.getenv("METERED_TURN_LABEL")
        required_env_vars = {
            "METERED_TURN_SECRET_KEY": self._metered_secret_key,
            "METERED_TURN_LABEL": self._metered_label,
        }
        missing_env_vars = [
            key for key, value in required_env_vars.items() if not value
        ]
        if missing_env_vars:
            missing = ", ".join(missing_env_vars)
            raise RuntimeError(
                f"Missing required Metered TURN env vars: {missing}. "
                "Set them on the server before starting the realtime app."
            )

        self._mode = "gta_drive"
        self._mode_seed_dirs = {
            "universal": "universal",
            "gta_drive": "gta_drive",
            "templerun": "temple_run",
        }
        seed_dir = self._mode_seed_dirs[self._mode]
        self._seed_path = self._matrix_repo_dir / f"demo_images/{seed_dir}/0000.png"
        if not self._seed_path.exists():
            raise RuntimeError(f"Seed image not found at {self._seed_path}")

        self._session_lock = threading.RLock()
        self._last_seed_key: str | None = None

        print("Warming up temporary session")
        warmup_session = None
        warmup_stream = None
        generated_frames = 0
        generated_blocks = 0
        try:
            warmup_session = self._build_session(mode=self._mode)
            warmup_session.prepare(str(self._seed_path), mode=self._mode)

            def action_provider(current_start_frame, num_frame_per_block, action_mode):
                return "q u"

            warmup_stream = warmup_session.stream_frames(action_provider)
            while True:
                block = next(warmup_stream)
                generated_frames += len(block)
                generated_blocks += 1
        except StopIteration:
            print(
                "Warmup complete: "
                f"{generated_frames} frames across {generated_blocks} blocks"
            )
        except Exception as exc:
            raise RuntimeError(f"Warmup failed: {exc}") from exc
        finally:
            if warmup_stream is not None:
                with suppress(Exception):
                    warmup_stream.close()

        self._session = self._build_session(mode=self._mode)

    def _build_ice_servers(self) -> list[dict]:
        import json
        import urllib.parse
        import urllib.request

        label = self._metered_label
        secret_key = self._metered_secret_key
        assert label is not None
        assert secret_key is not None
        credentials_url = f"https://{label}.metered.live/api/v1/turn/credentials"
        credential_url = f"https://{label}.metered.live/api/v1/turn/credential"

        def fetch_ice_servers(api_key: str) -> list[dict]:
            query = urllib.parse.urlencode({"apiKey": api_key})
            join_char = "&" if "?" in credentials_url else "?"
            url = f"{credentials_url}{join_char}{query}"
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = response.read().decode("utf-8")
            raw_servers = json.loads(payload)
            servers: list[dict] = []
            for item in raw_servers:
                urls = item.get("urls")
                if not urls:
                    continue
                servers.append(
                    {
                        "urls": urls,
                        "username": item.get("username"),
                        "credential": item.get("credential", item.get("password")),
                    }
                )
            return servers

        query = urllib.parse.urlencode({"secretKey": secret_key})
        join_char = "&" if "?" in credential_url else "?"
        url = f"{credential_url}{join_char}{query}"
        body = json.dumps(
            {
                "expiryInSeconds": self.TURN_EXPIRY_SECONDS,
                "label": "fal-webrtc-demo",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = response.read().decode("utf-8")
        credential_payload = json.loads(payload)
        temporary_api_key = credential_payload.get("apiKey")
        if not temporary_api_key:
            raise RuntimeError("Metered credential response missing apiKey.")

        servers = fetch_ice_servers(temporary_api_key)
        if not servers:
            raise RuntimeError("Metered returned empty ICE server list.")

        print("WebRTC: using Metered secret-minted ICE servers")
        return servers

    def _build_session(self, mode: str = "universal"):
        import glob
        import os
        import sys

        from inference_streaming import InteractiveGameStreamingSession, parse_args

        config_map = {
            "universal": "configs/inference_yaml/inference_universal.yaml",
            "gta_drive": "configs/inference_yaml/inference_gta_drive.yaml",
            "templerun": "configs/inference_yaml/inference_templerun.yaml",
        }
        config_path = config_map.get(mode, config_map["universal"])

        original_argv = sys.argv[:]
        try:
            sys.argv = [
                "matrix2.py",
                "--config_path",
                config_path,
                "--pretrained_model_path",
                "/data/Matrix-Game-2.0",
            ]
            args = parse_args()
        finally:
            sys.argv = original_argv

        if not args.checkpoint_path:
            config_name = os.path.basename(args.config_path)
            if config_name == "inference_universal.yaml":
                args.checkpoint_path = os.path.join(
                    args.pretrained_model_path,
                    "base_distilled_model",
                    "base_distill.safetensors",
                )
            elif config_name == "inference_templerun.yaml":
                args.checkpoint_path = os.path.join(
                    args.pretrained_model_path,
                    "templerun_distilled_model",
                    "templerun_7dim_onlykey.safetensors",
                )
            elif config_name == "inference_gta_drive.yaml":
                gta_candidates = glob.glob(
                    os.path.join(
                        args.pretrained_model_path, "**", "*gta*/*.safetensors"
                    ),
                    recursive=True,
                )
                if gta_candidates:
                    gta_candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
                    args.checkpoint_path = gta_candidates[0]
                else:
                    raise FileNotFoundError(
                        "No gta_drive checkpoint found under "
                        f"{args.pretrained_model_path}"
                    )
            else:
                candidates = glob.glob(
                    os.path.join(args.pretrained_model_path, "**", "*.safetensors"),
                    recursive=True,
                )
                if candidates:
                    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
                    args.checkpoint_path = candidates[0]

        if args.checkpoint_path:
            print(f"Using checkpoint: {args.checkpoint_path}")
        return InteractiveGameStreamingSession(args)

    def _prepare_session(self, force: bool = False):
        seed_key = f"{self._seed_path}:{self._mode}"
        with self._session_lock:
            if force:
                self._last_seed_key = None
            if self._last_seed_key != seed_key or not getattr(
                self._session, "_prepared", False
            ):
                self._session.prepare(str(self._seed_path), mode=self._mode)
                self._last_seed_key = seed_key
        return self._session

    @fal.realtime("/realtime", buffering=5)
    async def webrtc(
        self, inputs: AsyncIterator[RealtimeInput]
    ) -> AsyncIterator[RealtimeOutput]:
        import asyncio
        import concurrent.futures
        import time

        from aiortc import (
            RTCConfiguration,
            RTCIceServer,
            RTCPeerConnection,
            RTCSessionDescription,
            VideoStreamTrack,
        )
        from aiortc.sdp import candidate_from_sdp
        from av import VideoFrame

        session = await asyncio.to_thread(self._prepare_session, True)
        stream_holder = {"stream": None}
        state = {"last_action": "q u", "pending_action": None}
        render_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        action_queue: asyncio.Queue[object] = asyncio.Queue()
        metrics = {
            "generated_frames": 0,
            "sent_frames": 0,
            "generated_window_start": time.perf_counter(),
            "sent_window_start": time.perf_counter(),
        }

        def normalize_action(payload: object) -> str:
            movement_keys = {"w", "a", "s", "d", "q"}
            camera_keys = {"i", "j", "k", "l", "u"}
            aliases = {
                "up": "w",
                "down": "s",
                "left": "a",
                "right": "d",
            }

            if isinstance(payload, dict):
                value = payload.get("action")
            else:
                value = payload
            if value is None:
                return "q u"

            tokens = [token for token in str(value).strip().lower().split() if token]
            if not tokens:
                return "q u"

            move = "q"
            look = "u"
            for token in tokens:
                canonical = aliases.get(token, token)
                if canonical in movement_keys:
                    move = canonical
                if canonical in camera_keys:
                    look = canonical
            return f"{move} {look}"

        def action_provider(current_start_frame, num_frame_per_block, action_mode):
            if state["pending_action"] is not None:
                state["last_action"] = state["pending_action"]
                state["pending_action"] = None
            return state["last_action"]

        def reset_stream():
            state["pending_action"] = None
            state["last_action"] = "q u"
            old_stream = stream_holder.get("stream")
            close_stream = getattr(old_stream, "close", None)
            if callable(close_stream):
                with suppress(Exception):
                    close_stream()
            stream_holder["stream"] = None

        reset_stream()

        def render_block(action: str | None):
            with self._session_lock:
                stream = stream_holder["stream"]
                if stream is None:
                    stream = session.stream_frames(action_provider)
                    stream_holder["stream"] = stream
                if action is not None:
                    state["pending_action"] = action
                try:
                    return next(stream)
                except StopIteration:
                    return None

        async def render_block_async(action: str | None):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(render_executor, render_block, action)

        class BlockVideoTrack(VideoStreamTrack):
            def __init__(self, frame_queue):
                super().__init__()
                self._queue = frame_queue

            async def recv(self):
                frame = await self._queue.get()
                metrics["sent_frames"] += 1
                sent_elapsed = time.perf_counter() - float(metrics["sent_window_start"])
                if sent_elapsed >= 2.0:
                    send_fps = float(metrics["sent_frames"]) / sent_elapsed
                    print(
                        "WebRTC: send fps "
                        f"{send_fps:.1f} (queue={self._queue.qsize()})"
                    )
                    metrics["sent_frames"] = 0
                    metrics["sent_window_start"] = time.perf_counter()
                pts, time_base = await self.next_timestamp()
                frame.pts = pts
                frame.time_base = time_base
                return frame

        signal_ice_servers = self._build_ice_servers()
        rtc_ice_servers = [
            RTCIceServer(
                urls=server["urls"],
                username=server.get("username"),
                credential=server.get("credential"),
            )
            for server in signal_ice_servers
        ]
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=rtc_ice_servers)
        )
        frame_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=24)
        ready_for_frames = asyncio.Event()
        stop_event = asyncio.Event()
        resume_event = asyncio.Event()
        resume_event.set()
        outgoing: asyncio.Queue[RealtimeOutput | None] = asyncio.Queue()
        input_adapter = TypeAdapter(RealtimeInputMessage)

        async def send(payload: RealtimeOutputMessage) -> None:
            if stop_event.is_set():
                return
            await outgoing.put(RealtimeOutput(root=payload))

        async def send_error(prefix: str, exc: Exception) -> None:
            await send(
                ErrorOutput(type="error", error=f"{prefix}:{type(exc).__name__}:{exc}")
            )
            stop_event.set()
            await outgoing.put(None)

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate is None:
                await send(IceCandidateOutput(type="icecandidate", candidate=None))
                return
            await send(
                IceCandidateOutput(
                    type="icecandidate",
                    candidate=IceCandidate(
                        candidate=candidate.candidate,
                        sdpMid=candidate.sdpMid,
                        sdpMLineIndex=candidate.sdpMLineIndex,
                    ),
                )
            )

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"WebRTC: connection state {pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                stop_event.set()
                await outgoing.put(None)

        pc.addTrack(BlockVideoTrack(frame_queue))

        async def frame_producer():
            await ready_for_frames.wait()
            print("WebRTC: frame producer started")
            while not stop_event.is_set():
                await resume_event.wait()
                action_value: str | None = None
                try:
                    while True:
                        payload = action_queue.get_nowait()
                        action_value = normalize_action(payload)
                except asyncio.QueueEmpty:
                    pass
                if action_value is None:
                    action_value = state["last_action"]

                try:
                    started = time.perf_counter()
                    block = await render_block_async(action_value)
                    dt = time.perf_counter() - started
                    if dt > 0.5:
                        print(f"WebRTC: generated block in {dt:.2f}s")
                except Exception as exc:
                    await send_error("frame_failed", exc)
                    break

                if block is None:
                    await send(StreamExhaustedOutput(type="stream_exhausted"))
                    stop_event.set()
                    await outgoing.put(None)
                    break

                block_len = len(block)
                metrics["generated_frames"] += block_len
                generated_elapsed = time.perf_counter() - float(
                    metrics["generated_window_start"]
                )
                if generated_elapsed >= 2.0:
                    gen_fps = float(metrics["generated_frames"]) / generated_elapsed
                    print(
                        "WebRTC: generation fps "
                        f"{gen_fps:.1f} (block={block_len}, queue={frame_queue.qsize()})"
                    )
                    metrics["generated_frames"] = 0
                    metrics["generated_window_start"] = time.perf_counter()

                for frame in block:
                    video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
                    if frame_queue.full():
                        with suppress(asyncio.QueueEmpty):
                            frame_queue.get_nowait()
                    await frame_queue.put(video_frame)

        producer_task = asyncio.create_task(frame_producer())

        async def handle_offer(payload: OfferInput) -> bool:
            try:
                if pc.remoteDescription is not None:
                    print("WebRTC: ignoring duplicate offer")
                    return True
                offer = RTCSessionDescription(sdp=payload.sdp, type=payload.type)
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await send(AnswerOutput(type="answer", sdp=pc.localDescription.sdp))
                ready_for_frames.set()
                return True
            except Exception as exc:
                await send_error("offer_failed", exc)
                return False

        async def handle_icecandidate(payload: IceCandidateInput) -> bool:
            try:
                candidate = payload.candidate
                if candidate is None:
                    await pc.addIceCandidate(None)
                else:
                    parsed = candidate_from_sdp(candidate.candidate)
                    parsed.sdpMid = candidate.sdpMid
                    parsed.sdpMLineIndex = candidate.sdpMLineIndex
                    await pc.addIceCandidate(parsed)
                return True
            except Exception as exc:
                await send_error("ice_failed", exc)
                return False

        async def handle_action(payload: ActionInput) -> bool:
            await action_queue.put(payload.action)
            return True

        async def handle_pause(payload: PauseInput) -> bool:
            if payload.paused:
                resume_event.clear()
            else:
                resume_event.set()
            await send(PauseOutput(type="pause", paused=not resume_event.is_set()))
            return True

        async def handle_payload(payload: RealtimeInputMessage) -> bool:
            if isinstance(payload, OfferInput):
                return await handle_offer(payload)
            if isinstance(payload, IceCandidateInput):
                return await handle_icecandidate(payload)
            if isinstance(payload, ActionInput):
                return await handle_action(payload)
            if isinstance(payload, PauseInput):
                return await handle_pause(payload)
            return True

        async def input_loop() -> None:
            try:
                async for payload in inputs:
                    if stop_event.is_set():
                        break
                    try:
                        parsed_payload = (
                            payload.root
                            if isinstance(payload, RealtimeInput)
                            else input_adapter.validate_python(payload)
                        )
                    except ValidationError as exc:
                        await send_error("invalid_payload", exc)
                        break
                    should_continue = await handle_payload(parsed_payload)
                    if not should_continue:
                        break
            finally:
                stop_event.set()
                await outgoing.put(None)

        input_task: asyncio.Task | None = None
        try:
            await outgoing.put(
                RealtimeOutput(
                    root=IceServersOutput(
                        type="iceservers", iceservers=signal_ice_servers
                    )
                )
            )
            input_task = asyncio.create_task(input_loop())
            while True:
                payload = await outgoing.get()
                if payload is None:
                    break
                yield payload
        finally:
            print("WebRTC: session closing")
            stop_event.set()
            if input_task is not None:
                input_task.cancel()
                with suppress(asyncio.CancelledError):
                    await input_task
            producer_task.cancel()
            with suppress(asyncio.CancelledError):
                await producer_task
            reset_stream()
            with suppress(Exception):
                render_executor.shutdown(wait=False, cancel_futures=True)
            await pc.close()


if __name__ == "__main__":
    from matrix2_client import run

    info = MatrixWebRTC2.spawn()
    print(f"App ID: {info.application}")
    info.wait()
    run(endpoint=info.application + "/realtime")
