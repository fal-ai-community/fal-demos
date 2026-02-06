import argparse
import asyncio
import json
from contextlib import suppress

import fal_client
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp


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


def normalize_app_id(endpoint: str) -> str:
    normalized = endpoint.strip().strip("/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 2:
        raise ValueError(
            f"Invalid endpoint '{endpoint}'. Use <owner>/<app> or <owner>/<app>/webrtc."
        )
    if parts[-1] in {"realtime", "webrtc"}:
        parts = parts[:-1]
    return "/".join(parts[:2])


async def run_webrtc(*, endpoint: str):
    client = fal_client.AsyncClient()
    stop_event = asyncio.Event()
    paused = False

    app_id = normalize_app_id(endpoint)
    print(f"Connecting to realtime app {app_id}")
    async with client.realtime(app_id, use_jwt=False) as connection:
        pc = RTCPeerConnection()
        remote_queue: asyncio.Queue[object] = asyncio.Queue(maxsize=1)
        remote_tasks: list[asyncio.Task] = []

        async def rt_send(payload: dict) -> None:
            await connection.send(payload)

        async def rt_recv():
            return await connection.recv()

        @pc.on("icecandidate")
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

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state: {pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                stop_event.set()

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                print("Remote video track received.")
                remote_tasks.append(
                    asyncio.create_task(
                        forward_remote_frames(track, remote_queue, stop_event)
                    )
                )

        pc.addTransceiver("video", direction="recvonly")

        render_task = asyncio.create_task(
            render_frames("Matrix output", remote_queue, stop_event)
        )

        async def command_loop():
            nonlocal paused
            print(
                "Commands: w/a/s/d move, i/j/k/l look, pause, action <value>, "
                "raw <json>, quit"
            )
            while not stop_event.is_set():
                line = await asyncio.to_thread(input, "command> ")
                if line is None:
                    continue
                line = line.strip()
                if not line:
                    continue
                lower = line.lower()
                if lower in {"quit", "exit"}:
                    stop_event.set()
                    break
                if lower == "pause":
                    paused = not paused
                    await rt_send({"type": "pause", "paused": paused})
                    continue
                if lower in {"w", "a", "s", "d"}:
                    await rt_send({"type": "action", "action": f"{lower} u"})
                    continue
                if lower in {"i", "j", "k", "l"}:
                    await rt_send({"type": "action", "action": f"q {lower}"})
                    continue
                if lower.startswith("action "):
                    await rt_send({"type": "action", "action": line[7:].strip()})
                    continue
                if lower.startswith("raw "):
                    raw = line[4:].strip()
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        print(f"Invalid JSON: {exc}")
                        continue
                    await rt_send(payload)
                    continue
                print("Unknown command. Type 'quit' to exit.")

        command_task = asyncio.create_task(command_loop())

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await rt_send({"type": "offer", "sdp": pc.localDescription.sdp})
        await rt_send({"type": "action", "action": "q u"})
        print("Sent offer. Waiting for answer...")

        async def shutdown_on_stop():
            await stop_event.wait()
            with suppress(Exception):
                await connection.close()
            with suppress(Exception):
                await pc.close()

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
                    await pc.setRemoteDescription(
                        RTCSessionDescription(type="answer", sdp=msg["sdp"])
                    )
                elif msg_type == "icecandidate":
                    candidate = msg.get("candidate")
                    if candidate is None:
                        await pc.addIceCandidate(None)
                        continue
                    parsed = candidate_from_sdp(candidate.get("candidate", ""))
                    parsed.sdpMid = candidate.get("sdpMid")
                    parsed.sdpMLineIndex = candidate.get("sdpMLineIndex")
                    await pc.addIceCandidate(parsed)
                elif msg_type == "ready":
                    print("Server ready.")
                elif msg_type == "pause":
                    paused = bool(msg.get("paused", paused))
                    print(f"Paused: {paused}")
                elif msg_type == "stream_exhausted":
                    print("Stream exhausted.")
                elif msg_type == "error":
                    print(f"Server error: {msg.get('error')}")
                else:
                    print(f"WS message: {msg}")
        finally:
            stop_event.set()
            await connection.close()
            await pc.close()
            render_task.cancel()
            command_task.cancel()
            shutdown_task.cancel()
            for task in remote_tasks:
                task.cancel()
            with suppress(asyncio.CancelledError):
                await shutdown_task


def run(*args, **kwargs):
    print("Press 'q' in the output window or type 'quit' to stop.")
    try:
        asyncio.run(run_webrtc(*args, **kwargs))
    except KeyboardInterrupt:
        pass
    print("Done")


def main():
    parser = argparse.ArgumentParser(
        description="Connect to Matrix WebRTC realtime endpoint."
    )
    parser.add_argument(
        "--endpoint",
        help="Endpoint in the form <owner>/<app> or <owner>/<app>/webrtc",
    )
    args = parser.parse_args()

    run(endpoint=args.endpoint)


if __name__ == "__main__":
    main()
