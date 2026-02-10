import asyncio
from contextlib import suppress
from typing import Annotated, AsyncIterator, Literal

import fal
from fastapi import WebSocketDisconnect
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


RealtimeInputMessage = Annotated[
    OfferInput | IceCandidateInput,
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


class WebcamWebRtc(fal.App):
    machine_type = "GPU-H100"
    TURN_EXPIRY_SECONDS = 600
    requirements = [
        "aiortc",
        "av",
        "numpy",
        "opencv-python",
        "pydantic",
        "ultralytics",
    ]

    def setup(self):
        import os

        from ultralytics import YOLO

        model_path = "/data/yolov8n.pt"
        self.yolo_model = YOLO(model_path)
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
                "label": "fal-yolo-webrtc-demo",
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

    @fal.realtime("/realtime", buffering=5)
    async def webrtc(
        self, inputs: AsyncIterator[RealtimeInput]
    ) -> AsyncIterator[RealtimeOutput]:
        from aiortc import (
            RTCConfiguration,
            RTCIceServer,
            RTCPeerConnection,
            RTCSessionDescription,
        )
        from aiortc.contrib.media import MediaBlackhole
        from aiortc.sdp import candidate_from_sdp

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
        blackhole = MediaBlackhole()
        stop_event = asyncio.Event()
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
        async def on_connectionstatechange() -> None:
            if pc.connectionState in ("failed", "closed", "disconnected"):
                stop_event.set()
                await outgoing.put(None)

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                pc.addTrack(create_yolo_track(track, self.yolo_model))
            else:
                asyncio.ensure_future(blackhole.consume(track))

        async def handle_offer(payload: OfferInput) -> bool:
            try:
                offer = RTCSessionDescription(sdp=payload.sdp, type=payload.type)
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await send(AnswerOutput(type="answer", sdp=pc.localDescription.sdp))
                return True
            except Exception as exc:
                await send_error("offer_failed", exc)
                return False

        async def handle_icecandidate(payload: IceCandidateInput) -> bool:
            try:
                candidate = payload.candidate
                if candidate is None:
                    await pc.addIceCandidate(None)
                    return True
                parsed = candidate_from_sdp(candidate.candidate)
                parsed.sdpMid = candidate.sdpMid
                parsed.sdpMLineIndex = candidate.sdpMLineIndex
                await pc.addIceCandidate(parsed)
                return True
            except Exception as exc:
                await send_error("ice_failed", exc)
                return False

        async def handle_payload(payload: RealtimeInputMessage) -> bool:
            if isinstance(payload, OfferInput):
                return await handle_offer(payload)
            if isinstance(payload, IceCandidateInput):
                return await handle_icecandidate(payload)
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
                    raise WebSocketDisconnect()
                yield payload
        finally:
            stop_event.set()
            if input_task is not None:
                input_task.cancel()
                with suppress(asyncio.CancelledError):
                    await input_task
            await blackhole.stop()
            await pc.close()


def draw_boxes(image, detections):
    import cv2

    if detections is None:
        return image

    for box in detections.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        label = f"{detections.names.get(cls_id, 'obj')} {score:.2f}"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (int(x1), int(y1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return image


def create_yolo_track(source_track, yolo_model):
    from aiortc.mediastreams import MediaStreamTrack
    from av import VideoFrame

    class YOLOTrack(MediaStreamTrack):
        kind = "video"

        def __init__(self, track, model):
            super().__init__()
            self.source_track = track
            self.model = model

        async def recv(self):
            frame = await self.source_track.recv()
            img = frame.to_ndarray(format="bgr24")

            try:
                detections = self.model(img, verbose=False)[0]
                annotated = draw_boxes(img.copy(), detections)
            except Exception as exc:
                print(f"YOLO inference error: {exc}")
                annotated = img

            new_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

    return YOLOTrack(source_track, yolo_model)


if __name__ == "__main__":
    import asyncio

    from yolo_client import run

    info = WebcamWebRtc.spawn()
    print(f"App ID: {info.application}")
    info.wait()
    run(endpoint=info.application + "/realtime")
