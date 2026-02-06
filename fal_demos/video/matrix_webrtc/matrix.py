from contextlib import suppress
from typing import AsyncIterator

import fal
from fal.toolkit import clone_repository


class MatrixGame2(fal.App):
    machine_type = "GPU-H100"
    startup_timeout = 1200  # 20 minutes

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
        "huggingface_hub[cli]",
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
        import sys
        import threading

        self._repo_path = clone_repository(
            "https://github.com/efiop/Matrix-Game.git",
            commit_hash="edd9475d0600212d9205c6213543c5dd5c668b4f",
        )

        sys.path.insert(0, str(self._repo_path / "Matrix-Game-2"))
        os.chdir(self._repo_path / "Matrix-Game-2")
        os.system(
            "huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir /data/Matrix-Game-2.0"
        )

        self._default_mode = "universal"
        self._mode_seed_dirs = {
            "universal": "universal",
            "gta_drive": "gta_drive",
            "templerun": "temple_run",
        }
        seed_dir = self._mode_seed_dirs.get(self._default_mode, "universal")
        self._default_seed_path = (
            self._repo_path / f"Matrix-Game-2/demo_images/{seed_dir}/0001.png"
        )
        self._session_lock = threading.RLock()
        self._sessions = {
            self._default_mode: self._build_session(self._default_mode),
        }
        self._last_seed = {}

        session = self._prepare_session(True)
        with self._session_lock:

            def action_provider(current_start_frame, num_frame_per_block, action_mode):
                return "q u"

            stream = session.stream_frames(action_provider)
            try:
                next(stream)
            except StopIteration:
                pass
        self._last_seed[self._default_mode] = None

    def _build_session(self, mode="universal"):
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
        sys.argv = [
            "matrix_game2_app.py",
            "--config",
            config_path,
            "--pretrained_model_path",
            "/data/Matrix-Game-2.0",
        ]
        args = parse_args()
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
                        f"No gta_drive checkpoint found under {args.pretrained_model_path}"
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

    def _prepare_session(self, force=False):
        seed_image = str(self._default_seed_path)
        seed_key = f"{seed_image}:{self._default_mode}"
        with self._session_lock:
            session = self._sessions.get(self._default_mode)
            if session is None:
                raise ValueError(
                    f"Mode {self._default_mode} is not loaded in this deployment."
                )
            if force:
                self._last_seed[self._default_mode] = None
            if self._last_seed.get(self._default_mode) != seed_key or not getattr(
                session, "_prepared", False
            ):
                session.prepare(seed_image, mode=self._default_mode)
                self._last_seed[self._default_mode] = seed_key
        return session

    @fal.realtime("/webrtc")
    async def webrtc(self, inputs: AsyncIterator[dict]) -> AsyncIterator[dict]:
        import asyncio
        from av import VideoFrame
        from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
        from aiortc.sdp import candidate_from_sdp

        session = await asyncio.to_thread(self._prepare_session, True)
        stream_holder = {"stream": None}
        state = {"pending": None, "last": "q u"}

        def action_provider(current_start_frame, num_frame_per_block, action_mode):
            if state["pending"] is not None:
                state["last"] = state["pending"]
                state["pending"] = None
            return state["last"]

        def reset_stream():
            state["pending"] = None
            state["last"] = "q u"
            stream_holder["stream"] = session.stream_frames(action_provider)

        reset_stream()

        def render_block_frames(action):
            with self._session_lock:
                if action is not None:
                    state["pending"] = action
                try:
                    block = next(stream_holder["stream"])
                except StopIteration:
                    return None
            return block

        def coerce_action(payload):
            if isinstance(payload, dict):
                return payload.get("action", payload)
            return payload

        class BlockVideoTrack(VideoStreamTrack):
            def __init__(self, frame_queue):
                super().__init__()
                self._queue = frame_queue

            async def recv(self):
                frame = await self._queue.get()
                pts, time_base = await self.next_timestamp()
                frame.pts = pts
                frame.time_base = time_base
                return frame

        pc = RTCPeerConnection()
        frame_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=24)
        ready_for_frames = asyncio.Event()
        stop_event = asyncio.Event()
        resume_event = asyncio.Event()
        resume_event.set()
        first_frame_logged = False
        outgoing: asyncio.Queue[dict | None] = asyncio.Queue()

        async def send(payload: dict) -> None:
            if stop_event.is_set():
                return
            await outgoing.put(payload)

        async def send_error(prefix, exc):
            await send(
                {"type": "error", "error": f"{prefix}:{type(exc).__name__}:{exc}"}
            )
            stop_event.set()
            await outgoing.put(None)

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate is None:
                await send({"type": "icecandidate", "candidate": None})
                return
            await send(
                {
                    "type": "icecandidate",
                    "candidate": {
                        "candidate": candidate.candidate,
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex,
                    },
                }
            )

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"WebRTC: connection state {pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                stop_event.set()
                await outgoing.put(None)

        pc.addTrack(BlockVideoTrack(frame_queue))

        async def frame_producer():
            nonlocal session, first_frame_logged
            await ready_for_frames.wait()
            print("WebRTC: frame producer started")
            while not stop_event.is_set():
                await resume_event.wait()
                try:
                    payload = action_queue.get_nowait()
                except asyncio.QueueEmpty:
                    action = state["last"]
                else:
                    action = coerce_action(payload)
                    if action is not None:
                        state["last"] = action
                if action is None:
                    action = "q u"

                try:
                    block = await asyncio.to_thread(render_block_frames, action)
                except Exception as exc:
                    await send_error("frame_failed", exc)
                    break
                if block is None:
                    await send({"type": "stream_exhausted"})
                    stop_event.set()
                    await outgoing.put(None)
                    break

                for frame in block:
                    video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
                    await frame_queue.put(video_frame)
                    if not first_frame_logged:
                        print("WebRTC: enqueued first video frame")
                        first_frame_logged = True

        action_queue: asyncio.Queue[object] = asyncio.Queue()
        producer_task = asyncio.create_task(frame_producer())

        async def handle_offer(payload):
            try:
                print("WebRTC: received offer")
                offer_sdp = payload["sdp"]
                offer = RTCSessionDescription(sdp=offer_sdp, type=payload["type"])
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await send({"type": "answer", "sdp": pc.localDescription.sdp})
                ready_for_frames.set()
                return True
            except Exception as exc:
                await send_error("offer_failed", exc)
                return False

        async def handle_icecandidate(payload):
            try:
                print("WebRTC: received icecandidate")
                candidate = payload.get("candidate")
                if candidate is None:
                    await pc.addIceCandidate(None)
                else:
                    parsed = candidate_from_sdp(candidate.get("candidate", ""))
                    parsed.sdpMid = candidate.get("sdpMid")
                    parsed.sdpMLineIndex = candidate.get("sdpMLineIndex")
                    await pc.addIceCandidate(parsed)
                return True
            except Exception as exc:
                await send_error("ice_failed", exc)
                return False

        async def handle_action(payload):
            print(f"WebRTC: received action {payload.get('action')}")
            await action_queue.put(payload)
            return True

        async def handle_pause(payload):
            print(f"WebRTC: received pause {payload.get('paused')}")
            if payload.get("paused", False):
                resume_event.clear()
            else:
                resume_event.set()
            await send({"type": "pause", "paused": not resume_event.is_set()})
            return True

        async def handle_payload(payload):
            if isinstance(payload, dict):
                msg_type = payload.get("type")
                if msg_type == "offer":
                    return await handle_offer(payload)
                if msg_type == "icecandidate":
                    return await handle_icecandidate(payload)
                if msg_type == "action":
                    return await handle_action(payload)
                if msg_type == "pause":
                    return await handle_pause(payload)
            print(f"WebRTC: received raw payload {payload}")
            await action_queue.put(payload)
            return True

        async def input_loop() -> None:
            try:
                async for payload in inputs:
                    if stop_event.is_set():
                        break
                    should_continue = await handle_payload(payload)
                    if not should_continue:
                        break
            finally:
                stop_event.set()
                await outgoing.put(None)

        input_task: asyncio.Task | None = None
        try:
            await outgoing.put({"type": "ready"})
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
            await pc.close()
