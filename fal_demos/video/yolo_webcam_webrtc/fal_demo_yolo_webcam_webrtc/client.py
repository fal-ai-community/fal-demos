import argparse
import asyncio
from contextlib import suppress

import fal_client
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.sdp import candidate_from_sdp
from av import VideoFrame


class OpenCVCaptureTrack(VideoStreamTrack):
    def __init__(self, device, width, height, fps, preview_queue):
        super().__init__()
        import cv2

        self.cv2 = cv2
        self.preview_queue = preview_queue
        self.capture = cv2.VideoCapture(device)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open webcam device {device}")
        if width:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.capture.set(cv2.CAP_PROP_FPS, fps)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.capture.read()
        if not ret:
            await asyncio.sleep(0.02)
            return await self.recv()
        if self.preview_queue is not None:
            try:
                self.preview_queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    def stop(self):
        try:
            if self.capture:
                self.capture.release()
        finally:
            if self.preview_queue is not None:
                self.cv2.destroyWindow("Local webcam")
        super().stop()


async def forward_remote_frames(track, queue, stop_event):
    try:
        while not stop_event.is_set():
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            try:
                queue.put_nowait(img)
            except asyncio.QueueFull:
                pass
    except Exception as exc:
        print(f"Remote track error: {exc}")
    finally:
        stop_event.set()


async def render_frames(window_name, queue, stop_event, frame_shape=None):
    import cv2
    import numpy as np

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    placeholder = None
    if frame_shape:
        placeholder = np.zeros(frame_shape, dtype=np.uint8)

    try:
        while not stop_event.is_set():
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                frame = placeholder
            if frame is None:
                continue
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                stop_event.set()
                break
    except Exception as exc:
        print(f"Window error ({window_name}): {exc}")
    finally:
        cv2.destroyWindow(window_name)


def parse_device(value):
    if value is None:
        return 0
    try:
        return int(value)
    except ValueError:
        return value


def normalize_app_id(endpoint: str) -> str:
    normalized = endpoint.strip().strip("/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 2:
        raise ValueError(
            f"Invalid endpoint '{endpoint}'. Use <owner>/<app> or <owner>/<app>/realtime."
        )
    if parts[-1] in {"realtime", "webrtc"}:
        parts = parts[:-1]
    return "/".join(parts[:2])


def parse_ice_servers(ice_servers_payload: object) -> list[RTCIceServer]:
    if not isinstance(ice_servers_payload, list):
        return [RTCIceServer(urls="stun:stun.l.google.com:19302")]

    servers: list[RTCIceServer] = []
    for item in ice_servers_payload:
        if not isinstance(item, dict):
            continue
        urls = item.get("urls")
        if not urls:
            continue
        servers.append(
            RTCIceServer(
                urls=urls,
                username=item.get("username"),
                credential=item.get("credential"),
            )
        )
    if servers:
        return servers
    return [RTCIceServer(urls="stun:stun.l.google.com:19302")]


async def run_webrtc(
    *,
    endpoint: str,
    device: int | str = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    no_preview: bool = False,
):
    client = fal_client.AsyncClient()
    stop_event = asyncio.Event()

    app_id = normalize_app_id(endpoint)
    print(f"Connecting to realtime app {app_id}")
    async with client.realtime(
        app_id,
        path="/realtime",
        use_jwt=False,
    ) as connection:
        pc: RTCPeerConnection | None = None
        local_preview_queue = None if no_preview else asyncio.Queue(maxsize=1)
        remote_queue = asyncio.Queue(maxsize=1)
        webcam = OpenCVCaptureTrack(
            device=device,
            width=width,
            height=height,
            fps=fps,
            preview_queue=local_preview_queue,
        )

        async def rt_send(payload):
            await connection.send(payload)

        async def rt_recv():
            return await connection.recv()

        def create_peer_connection(ice_servers_payload: object) -> RTCPeerConnection:
            ice_servers = parse_ice_servers(ice_servers_payload)
            print(f"Using {len(ice_servers)} ICE server entries from signaling.")
            local_pc = RTCPeerConnection(
                configuration=RTCConfiguration(iceServers=ice_servers)
            )
            local_pc.addTrack(webcam)

            @local_pc.on("icecandidate")
            async def on_icecandidate(candidate):
                if candidate is None:
                    await rt_send({"type": "icecandidate", "candidate": None})
                    return
                await rt_send(
                    {
                        "type": "icecandidate",
                        "candidate": {
                            "candidate": candidate.candidate,
                            "sdpMid": candidate.sdpMid,
                            "sdpMLineIndex": candidate.sdpMLineIndex,
                        },
                    }
                )

            @local_pc.on("connectionstatechange")
            async def on_connectionstatechange():
                print(f"Connection state: {local_pc.connectionState}")
                if local_pc.connectionState in ("failed", "closed", "disconnected"):
                    stop_event.set()

            @local_pc.on("track")
            def on_track(track):
                if track.kind == "video":
                    print("Remote video track received.")
                    asyncio.create_task(
                        forward_remote_frames(track, remote_queue, stop_event)
                    )

            return local_pc

        render_tasks = []
        if local_preview_queue is not None:
            render_tasks.append(
                asyncio.create_task(
                    render_frames(
                        "Local webcam",
                        local_preview_queue,
                        stop_event,
                        frame_shape=(height, width, 3),
                    )
                )
            )
        render_tasks.append(
            asyncio.create_task(render_frames("YOLO output", remote_queue, stop_event))
        )

        offer_sent = False

        async def shutdown_on_stop():
            await stop_event.wait()
            with suppress(Exception):
                await connection.close()
            if pc is not None:
                with suppress(Exception):
                    await pc.close()
            with suppress(Exception):
                webcam.stop()

        shutdown_task = asyncio.create_task(shutdown_on_stop())

        try:
            while not stop_event.is_set():
                msg = await rt_recv()
                if msg is None:
                    print("Server closed connection.")
                    break
                if not isinstance(msg, dict):
                    print(f"WS message: {msg}")
                    continue
                msg_type = msg.get("type")
                if msg_type == "answer" and msg.get("sdp"):
                    if pc is None:
                        continue
                    await pc.setRemoteDescription(
                        RTCSessionDescription(type="answer", sdp=msg["sdp"])
                    )
                elif msg_type == "iceservers" and not offer_sent:
                    pc = create_peer_connection(msg.get("iceservers"))
                    offer = await pc.createOffer()
                    await pc.setLocalDescription(offer)
                    await rt_send({"type": "offer", "sdp": pc.localDescription.sdp})
                    offer_sent = True
                    print("Sent offer. Waiting for answer...")
                elif msg_type == "icecandidate":
                    if pc is None:
                        continue
                    candidate = msg.get("candidate")
                    if candidate is None:
                        await pc.addIceCandidate(None)
                        continue
                    parsed = candidate_from_sdp(candidate.get("candidate", ""))
                    parsed.sdpMid = candidate.get("sdpMid")
                    parsed.sdpMLineIndex = candidate.get("sdpMLineIndex")
                    await pc.addIceCandidate(parsed)
                elif msg_type == "error":
                    print(f"Server error: {msg.get('error')}")
                else:
                    print(f"WS message: {msg}")
        finally:
            stop_event.set()
            await connection.close()
            if pc is not None:
                await pc.close()
            webcam.stop()
            for task in render_tasks:
                task.cancel()
            shutdown_task.cancel()
            with suppress(asyncio.CancelledError):
                await shutdown_task


def run(*args, **kwargs):
    print("Press 'q' in the output window to stop.")
    try:
        asyncio.run(run_webrtc(*args, **kwargs))
    except KeyboardInterrupt:
        pass
    print("Done")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stream webcam to a fal realtime endpoint and display the YOLO output."
        )
    )
    parser.add_argument(
        "--endpoint",
        help="Endpoint in the form <owner>/<app> or <owner>/<app>/realtime",
    )
    parser.add_argument(
        "--device",
        type=parse_device,
        default=0,
        help="Webcam device index or path (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Webcam capture width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Webcam capture height (default: 480)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Webcam capture FPS (default: 30)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable local webcam preview window",
    )
    args = parser.parse_args()

    run(
        endpoint=args.endpoint,
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        no_preview=args.no_preview,
    )


if __name__ == "__main__":
    main()
