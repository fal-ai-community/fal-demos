import asyncio
import json
import os

import fal
from fastapi import WebSocket


class WebcamWebRtc(fal.App):
    machine_type = "GPU-H100"
    requirements = [
        "aiortc",
        "av",
        "numpy",
        "opencv-python",
        "ultralytics",
    ]

    def setup(self):
        from ultralytics import YOLO
    
        model_path = "/data/yolov8n.pt"
        self.yolo_model = YOLO(model_path)

    @fal.endpoint("/webrtc", is_websocket=True)
    async def webrtc(self, ws: WebSocket):
        from aiortc import RTCPeerConnection, RTCSessionDescription
        from aiortc.contrib.media import MediaBlackhole
        from aiortc.sdp import candidate_from_sdp
        from starlette.websockets import WebSocketDisconnect, WebSocketState

        await ws.accept()
        await ws.send_json({"type": "ready"})

        pc = RTCPeerConnection()
        blackhole = MediaBlackhole()
        stop_event = asyncio.Event()

        async def safe_send_json(payload):
            try:
                if (
                    ws.client_state != WebSocketState.CONNECTED
                    or ws.application_state != WebSocketState.CONNECTED
                ):
                    return
                await ws.send_json(payload)
            except (RuntimeError, WebSocketDisconnect):
                pass

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate is None:
                await safe_send_json({"type": "icecandidate", "candidate": None})
                return
            await safe_send_json(
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
            if pc.connectionState in ("failed", "closed", "disconnected"):
                stop_event.set()

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                pc.addTrack(create_yolo_track(track, self.yolo_model))
            else:
                asyncio.ensure_future(blackhole.consume(track))

        async def handle_offer(payload):
            offer = RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await safe_send_json({"type": "answer", "sdp": pc.localDescription.sdp})
            return True

        async def handle_icecandidate(payload):
            candidate = payload.get("candidate")
            if candidate is None:
                await pc.addIceCandidate(None)
                return True
            parsed = candidate_from_sdp(candidate.get("candidate", ""))
            parsed.sdpMid = candidate.get("sdpMid")
            parsed.sdpMLineIndex = candidate.get("sdpMLineIndex")
            await pc.addIceCandidate(parsed)
            return True

        async def handle_payload(payload):
            if isinstance(payload, dict):
                msg_type = payload.get("type")
                if msg_type == "offer":
                    return await handle_offer(payload)
                if msg_type == "icecandidate":
                    return await handle_icecandidate(payload)
            return True

        async def receive_payload():
            try:
                message = await ws.receive_text()
            except RuntimeError:
                return None
            try:
                return json.loads(message)
            except json.JSONDecodeError:
                return message

        try:
            while not stop_event.is_set():
                payload = await receive_payload()
                if payload is None:
                    break
                should_continue = await handle_payload(payload)
                if not should_continue:
                    break
        except WebSocketDisconnect:
            pass
        finally:
            stop_event.set()
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
