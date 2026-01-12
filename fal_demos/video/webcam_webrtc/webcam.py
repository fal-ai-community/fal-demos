import asyncio
import json

import fal
from fastapi import WebSocket


class WebcamWebRtc(fal.App):
    requirements = [
        "aiortc",
        "av",
    ]

    @fal.endpoint("/webrtc", is_websocket=True)
    async def webrtc(self, ws: WebSocket):
        from aiortc import RTCPeerConnection, RTCSessionDescription
        from aiortc.contrib.media import MediaBlackhole
        from aiortc.mediastreams import MediaStreamTrack
        from aiortc.sdp import candidate_from_sdp
        from starlette.websockets import WebSocketDisconnect, WebSocketState

        class PassthroughVideoTrack(MediaStreamTrack):
            kind = "video"

            def __init__(self, source_track):
                super().__init__()
                self.source_track = source_track

            async def recv(self):
                return await self.source_track.recv()

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
                pc.addTrack(PassthroughVideoTrack(track))
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
