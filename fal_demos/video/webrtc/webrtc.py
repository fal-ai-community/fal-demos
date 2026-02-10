from contextlib import suppress
from typing import Annotated, AsyncIterator, Literal

from pydantic import BaseModel, Field, RootModel, TypeAdapter, ValidationError
import fal


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


RealtimeInputMessage = Annotated[
    OfferInput | IceCandidateInput | ActionInput,
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


class ErrorOutput(BaseModel):
    type: Literal["error"]
    error: str


RealtimeOutputMessage = Annotated[
    IceServersOutput | AnswerOutput | IceCandidateOutput | ErrorOutput,
    Field(discriminator="type"),
]


class RealtimeOutput(RootModel):
    root: RealtimeOutputMessage


class BasicWebRTCVideo(fal.App):
    machine_type = "M"
    TURN_EXPIRY_SECONDS = 600
    requirements = [
        "aiortc",
        "av",
        "numpy",
        "opencv-python>=4.9.0.80",
    ]

    def setup(self):
        import os

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

    def _make_frame(
        self,
        *,
        last_key: str,
        animation: dict,
        fps_value: float,
        message_templates: list[str],
    ):
        import colorsys

        import cv2
        import numpy as np
        from av import VideoFrame

        height, width = 480, 640
        hue = (animation["frame_index"] * 0.0035) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        bgr = np.array([int(b * 255), int(g * 255), int(r * 255)], dtype=np.uint8)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = bgr

        frame_index = int(animation["frame_index"])
        template = message_templates[(frame_index // 75) % len(message_templates)]
        text = template.format(key=last_key)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 1.0
        text_thickness = 2
        text_lines = text.splitlines()
        line_metrics = [
            cv2.getTextSize(line, font, text_scale, text_thickness)
            for line in text_lines
        ]
        line_widths = [size[0][0] for size in line_metrics]
        line_heights = [size[0][1] for size in line_metrics]
        max_baseline = max(size[1] for size in line_metrics)
        line_step = max(line_heights) + 10
        text_width = max(line_widths)
        margin = 8
        min_x = float(margin)
        max_x = float(width - margin - text_width)
        min_y = float(margin + line_heights[0])
        max_y = float(
            height - margin - max_baseline - (len(text_lines) - 1) * line_step
        )

        if animation["x"] <= min_x or animation["x"] >= max_x:
            animation["vx"] *= -1.0
        if animation["y"] <= min_y or animation["y"] >= max_y:
            animation["vy"] *= -1.0

        animation["x"] = min(max(animation["x"] + animation["vx"], min_x), max_x)
        animation["y"] = min(max(animation["y"] + animation["vy"], min_y), max_y)

        line_x = int(animation["x"])
        line_y = int(animation["y"])
        for line in text_lines:
            cv2.putText(
                frame,
                line,
                (line_x, line_y),
                font,
                text_scale,
                (0, 0, 255),
                text_thickness,
                cv2.LINE_AA,
            )
            line_y += line_step
        fps_text = f"FPS: {fps_value:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (12, 30),
            font,
            0.75,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            fps_text,
            (12, 30),
            font,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        animation["frame_index"] += 1
        return VideoFrame.from_ndarray(frame, format="bgr24")

    @fal.realtime("/realtime")
    async def webrtc(
        self, inputs: AsyncIterator[RealtimeInput]
    ) -> AsyncIterator[RealtimeOutput]:
        import asyncio
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

        state = {"last_key": "none"}
        animation = {"x": 24.0, "y": 240.0, "vx": 3.2, "vy": 2.4, "frame_index": 0}
        fps_state = {"value": 0.0, "last_ts": None}
        message_templates = [
            "You pressed {key}\n[Wow]",
            "You pressed {key}\n[Much webrtc]",
            "You pressed {key}\n[Such realtime]",
        ]

        class GeneratedVideoTrack(VideoStreamTrack):
            def __init__(self, frame_queue):
                super().__init__()
                self._queue = frame_queue

            async def recv(self):
                frame = await self._queue.get()
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
        ice_config = RTCConfiguration(iceServers=rtc_ice_servers)
        pc = RTCPeerConnection(configuration=ice_config)
        frame_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=3)
        ready_for_frames = asyncio.Event()
        stop_event = asyncio.Event()
        outgoing: asyncio.Queue[RealtimeOutput | None] = asyncio.Queue()
        input_adapter = TypeAdapter(RealtimeInputMessage)

        async def send(payload: RealtimeOutputMessage) -> None:
            if stop_event.is_set():
                return
            await outgoing.put(RealtimeOutput(root=payload))

        async def send_error(prefix, exc):
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

        pc.addTrack(GeneratedVideoTrack(frame_queue))

        async def frame_producer():
            await ready_for_frames.wait()
            while not stop_event.is_set():
                now = time.perf_counter()
                if fps_state["last_ts"] is not None:
                    dt = now - float(fps_state["last_ts"])
                    if dt > 0:
                        instant_fps = 1.0 / dt
                        if fps_state["value"] == 0.0:
                            fps_state["value"] = instant_fps
                        else:
                            fps_state["value"] = (0.9 * fps_state["value"]) + (
                                0.1 * instant_fps
                            )
                fps_state["last_ts"] = now
                video_frame = self._make_frame(
                    last_key=str(state["last_key"]),
                    animation=animation,
                    fps_value=float(fps_state["value"]),
                    message_templates=message_templates,
                )
                if frame_queue.full():
                    with suppress(asyncio.QueueEmpty):
                        frame_queue.get_nowait()
                await frame_queue.put(video_frame)
                await asyncio.sleep(1 / 30)

        producer_task = asyncio.create_task(frame_producer())

        async def handle_offer(payload: OfferInput):
            try:
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

        async def handle_icecandidate(payload: IceCandidateInput):
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

        async def handle_action(payload: ActionInput):
            state["last_key"] = str(payload.action)
            return True

        async def handle_payload(payload: RealtimeInputMessage):
            if isinstance(payload, OfferInput):
                return await handle_offer(payload)
            if isinstance(payload, IceCandidateInput):
                return await handle_icecandidate(payload)
            if isinstance(payload, ActionInput):
                return await handle_action(payload)
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
            stop_event.set()
            if input_task is not None:
                input_task.cancel()
                with suppress(asyncio.CancelledError):
                    await input_task
            producer_task.cancel()
            with suppress(asyncio.CancelledError):
                await producer_task
            await pc.close()


if __name__ == "__main__":
    from webrtc_client import run

    info = BasicWebRTCVideo.spawn()
    print(f"App ID: {info.application}")
    info.wait()
    run(endpoint=info.application + "/realtime")
