import argparse
import asyncio
from contextlib import suppress

import fal_client
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
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


def keycode_to_action(key_code: int) -> str | None:
    if not (0 <= key_code < 256):
        return None
    key = chr(key_code).lower()
    if key in {"w", "a", "s", "d"}:
        return f"{key} u"
    if key in {"i", "j", "k", "l"}:
        return f"q {key}"
    return None


async def render_and_send_controls(
    window_name: str,
    queue: asyncio.Queue,
    stop_event: asyncio.Event,
    send_action,
    send_pause,
):
    import cv2
    import numpy as np

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    placeholder = np.zeros((352, 640, 3), dtype=np.uint8)
    paused = False

    try:
        while not stop_event.is_set():
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=0.03)
            except asyncio.TimeoutError:
                frame = placeholder

            cv2.imshow(window_name, frame)
            key_code = cv2.waitKeyEx(1)
            if key_code in (ord("q"), ord("Q")):
                stop_event.set()
                break

            if key_code == 32:  # Space
                paused = not paused
                await send_pause(paused)
                print(f"Paused: {paused}")
                continue

            action = keycode_to_action(key_code)
            if action:
                await send_action(action)
    except Exception as exc:
        print(f"Window error ({window_name}): {exc}")
    finally:
        cv2.destroyWindow(window_name)


async def run_webrtc(*, endpoint: str):
    client = fal_client.AsyncClient()
    stop_event = asyncio.Event()

    app_id = normalize_app_id(endpoint)
    print(f"Connecting to realtime app {app_id}")
    async with client.realtime(app_id, path="/realtime", use_jwt=False) as connection:
        pc: RTCPeerConnection | None = None
        remote_queue: asyncio.Queue[object] = asyncio.Queue(maxsize=1)
        remote_tasks: list[asyncio.Task] = []

        async def rt_send(payload: dict) -> None:
            await connection.send(payload)

        async def rt_recv():
            return await connection.recv()

        async def send_action(action_text: str) -> None:
            await rt_send({"type": "action", "action": action_text})

        async def send_pause(paused: bool) -> None:
            await rt_send({"type": "pause", "paused": paused})

        def create_peer_connection(ice_servers_payload: object) -> RTCPeerConnection:
            ice_servers = parse_ice_servers(ice_servers_payload)
            print(f"Using {len(ice_servers)} ICE server entries from signaling.")
            local_pc = RTCPeerConnection(
                configuration=RTCConfiguration(iceServers=ice_servers)
            )

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
                    remote_tasks.append(
                        asyncio.create_task(
                            forward_remote_frames(track, remote_queue, stop_event)
                        )
                    )

            local_pc.addTransceiver("video", direction="recvonly")
            return local_pc

        render_task = asyncio.create_task(
            render_and_send_controls(
                "Matrix2 output (WASD move, IJKL look, Space pause)",
                remote_queue,
                stop_event,
                send_action,
                send_pause,
            )
        )

        offer_sent = False

        async def shutdown_on_stop():
            await stop_event.wait()
            with suppress(Exception):
                await connection.close()
            if pc is not None:
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
                if msg_type == "iceservers" and not offer_sent:
                    pc = create_peer_connection(msg.get("iceservers"))
                    offer = await pc.createOffer()
                    await pc.setLocalDescription(offer)
                    await rt_send({"type": "offer", "sdp": pc.localDescription.sdp})
                    await send_action("q u")
                    offer_sent = True
                    print("Sent offer. Waiting for answer...")
                elif msg_type == "answer" and msg.get("sdp") and pc is not None:
                    await pc.setRemoteDescription(
                        RTCSessionDescription(type="answer", sdp=msg["sdp"])
                    )
                elif msg_type == "icecandidate" and pc is not None:
                    candidate = msg.get("candidate")
                    if candidate is None:
                        await pc.addIceCandidate(None)
                        continue
                    parsed = candidate_from_sdp(candidate.get("candidate", ""))
                    parsed.sdpMid = candidate.get("sdpMid")
                    parsed.sdpMLineIndex = candidate.get("sdpMLineIndex")
                    await pc.addIceCandidate(parsed)
                elif msg_type == "pause":
                    print(f"Paused: {bool(msg.get('paused', False))}")
                elif msg_type == "error":
                    print(f"Server error: {msg.get('error')}")
                elif msg_type == "stream_exhausted":
                    print("Stream exhausted.")
                else:
                    print(f"WS message: {msg}")
        finally:
            stop_event.set()
            await connection.close()
            if pc is not None:
                await pc.close()
            render_task.cancel()
            shutdown_task.cancel()
            for task in remote_tasks:
                task.cancel()
            with suppress(asyncio.CancelledError):
                await shutdown_task


def run(*args, **kwargs):
    print("Focus the video window. WASD move, IJKL look, Space pause, q quits.")
    try:
        asyncio.run(run_webrtc(*args, **kwargs))
    except KeyboardInterrupt:
        pass
    print("Done")


def main():
    parser = argparse.ArgumentParser(
        description="Connect to Matrix2 WebRTC realtime endpoint."
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Endpoint in the form <owner>/<app> or <owner>/<app>/realtime",
    )
    args = parser.parse_args()
    run(endpoint=args.endpoint)


if __name__ == "__main__":
    main()
