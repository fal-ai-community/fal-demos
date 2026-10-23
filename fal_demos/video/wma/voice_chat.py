"""Hold a live spoken conversation with Moshi over fal WMA.

The browser sends its microphone track. The app resamples the audio to 24 kHz
mono, feeds it to Moshi (Kyutai, 7B, full duplex) in 80 ms frames, and sends
Moshi's voice back as an audio track. Moshi listens and speaks at the same
time, so the caller can interrupt it. Each text token that Moshi speaks is sent
as a `text` message, and a `reset` control message starts a new conversation.
"""

import asyncio
import fractions
import logging
import threading
import time
from collections import deque
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

import fal.wma

logger = logging.getLogger(__name__)

# Weights, pinned to one revision (CC-BY-4.0). The moshi package is MIT and Apache-2.0.
MOSHI_REPO = "kyutai/moshiko-pytorch-bf16"
MOSHI_REVISION = "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"

SAMPLE_RATE = 24_000  # Mimi codec rate, for both input and output.
FRAME_SAMPLES = 1920  # One Mimi frame: 80 ms at 24 kHz (12.5 frames per second).
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
PACKET_SAMPLES = 480  # 20 ms per outbound WebRTC audio frame.
MAX_BACKLOG_FRAMES = 4  # Waiting input frames (320 ms). Older ones are dropped.
OUTPUT_QUEUE_FRAMES = 25  # Generated frames waiting to play (2 s). Older ones are dropped.
# Moshi's text tokens 0 and 3 are padding, not words.
TEXT_PADDING_TOKENS = (0, 3)
MAX_SESSION_SECONDS = 300


class Reset(BaseModel):
    type: Literal["reset"] = "reset"


class Text(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(description="One piece of what Moshi is saying, in order.")


class Status(BaseModel):
    type: Literal["status"] = "status"
    state: Literal["waiting_for_microphone", "listening", "restarting"]
    heard_seconds: float = 0.0
    spoken_seconds: float = 0.0
    dropped_input_frames: int = 0


class Rejected(BaseModel):
    type: Literal["rejected"] = "rejected"
    reason: str


ServerMessage = Annotated[Union[Text, Status, Rejected], Field(discriminator="type")]


class Moshi:
    """Mimi codec, Moshi language model, and text tokenizer in streaming mode."""

    def __init__(self):
        import torch
        from moshi.models import LMGen, loaders

        torch.set_grad_enabled(False)
        info = loaders.CheckpointInfo.from_hf_repo(MOSHI_REPO, revision=MOSHI_REVISION)
        self.mimi = info.get_mimi(device="cuda")
        self.tokenizer = info.get_text_tokenizer()
        lm = info.get_moshi(device="cuda", dtype=torch.bfloat16)
        self.lm_gen = LMGen(lm, **info.lm_gen_config)
        # Streaming state lives for the whole runner and is reset per conversation.
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self.warm_up()

    def warm_up(self):
        """Run a few silent frames so CUDA graphs are captured before the first session."""
        import torch

        for _ in range(4):
            self.step(torch.zeros(1, 1, FRAME_SAMPLES, device="cuda"))
        torch.cuda.synchronize()
        self.reset()

    def reset(self):
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    def step(self, chunk, skip=False):
        """Feed one input frame. Return (pcm, text) pairs for the frames Moshi produced.

        `chunk` is float32 audio shaped (1, 1, FRAME_SAMPLES). With `skip`, the
        frame is encoded and then discarded, as upstream does for the first frame.
        """
        codes = self.mimi.encode(chunk)
        if skip:
            # The first encoded frame carries the encoder's left padding. Upstream
            # resets the encoder so the next frame gets the padding again.
            self.mimi.reset_streaming()
        out = []
        for c in range(codes.shape[-1]):
            tokens = self.lm_gen.step(codes[:, :, c : c + 1])
            if tokens is None:  # Moshi has not produced a frame yet.
                continue
            pcm = self.mimi.decode(tokens[:, 1:])[0, 0].float().cpu().numpy()
            token = tokens[0, 0, 0].item()
            text = None
            if token not in TEXT_PADDING_TOKENS:
                text = self.tokenizer.id_to_piece(token).replace("▁", " ")
            out.append((pcm, text))
        return out


def make_audio_queue_track(queue):
    """Build an outbound audio track that plays float32 PCM chunks from `queue`.

    The track paces 20 ms packets against a wall clock and plays silence while
    the queue is empty, so the caller hears a steady stream.
    """
    import av
    import numpy as np
    from aiortc import MediaStreamTrack

    class AudioQueueTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self):
            super().__init__()
            self.pending = np.zeros(0, dtype=np.float32)
            self.pts = 0
            self.start = None

        def queued_samples(self):
            return len(self.pending) + sum(len(chunk) for chunk in queue._queue)

        async def recv(self):
            now = time.monotonic()
            if self.start is None:
                self.start = now
            due = self.start + self.pts / SAMPLE_RATE
            if due < now - 1:  # The consumer stalled. Do not burst to catch up.
                self.start = now - self.pts / SAMPLE_RATE
            elif due > now:
                await asyncio.sleep(due - now)
            while len(self.pending) < PACKET_SAMPLES and not queue.empty():
                self.pending = np.concatenate([self.pending, queue.get_nowait()])
            if len(self.pending) >= PACKET_SAMPLES:
                chunk = self.pending[:PACKET_SAMPLES]
                self.pending = self.pending[PACKET_SAMPLES:]
            else:
                chunk = np.zeros(PACKET_SAMPLES, dtype=np.float32)
            pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
            frame = av.AudioFrame.from_ndarray(pcm[None, :], format="s16", layout="mono")
            frame.sample_rate = SAMPLE_RATE
            frame.pts = self.pts
            frame.time_base = fractions.Fraction(1, SAMPLE_RATE)
            self.pts += PACKET_SAMPLES
            return frame

    return AudioQueueTrack()


class VoiceSession:
    """Per-session buffers and the thread that runs the model."""

    def __init__(self, app, session, loop):
        self.app = app
        self.session = session
        self.loop = loop
        self.state = "waiting_for_microphone"
        self.frames = deque()  # Input frames, oldest first.
        self.out = asyncio.Queue(maxsize=OUTPUT_QUEUE_FRAMES)
        self.track = None  # The outbound audio track.
        self.cond = threading.Condition()
        self.stopped = threading.Event()
        self.reset_requested = False
        self.thread = None
        self.heard_frames = 0
        self.spoken_frames = 0
        self.dropped_input_frames = 0
        self.dropped_output_samples = 0  # Billed audio that never played. Loop only.

    # Event loop side: control messages and microphone input.

    def on_reset(self, message):
        try:
            Reset.model_validate(message)
        except ValueError:
            self.session.send(Rejected(reason="invalid reset message").model_dump())
            return
        with self.cond:
            self.reset_requested = True
            self.frames.clear()
            self.cond.notify_all()

    def start(self, microphone, track):
        self.track = track
        self.session.create_task(self.read_microphone(microphone))
        self.session.create_task(self.report_status())
        self.thread = threading.Thread(target=self.run, name="moshi", daemon=True)
        self.thread.start()

    async def read_microphone(self, microphone):
        import av
        import numpy as np

        # Browsers send 48 kHz Opus. Moshi expects 24 kHz mono float32.
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        buffer = np.zeros(0, dtype=np.float32)
        self.set_state("listening")
        while True:
            try:
                frame = await microphone.recv()
            except Exception:
                break
            for resampled in resampler.resample(frame):
                buffer = np.concatenate([buffer, resampled.to_ndarray()[0]])
            with self.cond:
                while len(buffer) >= FRAME_SAMPLES:
                    self.frames.append(buffer[:FRAME_SAMPLES])
                    buffer = buffer[FRAME_SAMPLES:]
                    while len(self.frames) > MAX_BACKLOG_FRAMES:
                        self.frames.popleft()
                        self.dropped_input_frames += 1
                self.cond.notify_all()
        # The microphone ended. Stop feeding frames so the model stops generating.
        with self.cond:
            self.frames.clear()
        self.set_state("waiting_for_microphone")

    async def report_status(self):
        while True:
            await asyncio.sleep(2)
            self.send_status()

    def send_status(self):
        self.session.send(
            Status(
                state=self.state,
                heard_seconds=round(self.heard_frames * FRAME_SECONDS, 2),
                spoken_seconds=round(self.spoken_frames * FRAME_SECONDS, 2),
                dropped_input_frames=self.dropped_input_frames,
            ).model_dump()
        )

    def put_output(self, pcm):
        """Queue audio for playback on the event loop. Count the audio it displaces."""
        if self.out.full():
            self.dropped_output_samples += len(self.out._queue[0])
        fal.wma.queue_put_drop_oldest(self.out, pcm)

    async def cap_billing(self):
        """Stop billing for audio that was billed but never reached playback.

        Runs after the model thread has stopped and before billing is finalized.
        """
        await asyncio.sleep(0)  # Let queued put_output callbacks run first.
        undelivered = self.dropped_output_samples
        if self.track is not None:
            undelivered += self.track.queued_samples()
        self.session.cap_billable_units(
            max(0.0, self.session.billable_units - undelivered / SAMPLE_RATE)
        )

    async def shutdown(self):
        self.stopped.set()
        with self.cond:
            self.cond.notify_all()
        if self.thread is not None:
            await asyncio.to_thread(self.thread.join, 30)

    # Model thread.

    def next_frame(self):
        """Return the next input frame, "reset" if a reset was requested, None if stopped."""
        with self.cond:
            while not self.frames and not self.reset_requested:
                if self.stopped.is_set():
                    return None
                self.cond.wait(timeout=0.5)
            if self.stopped.is_set():
                return None
            if self.reset_requested:
                self.reset_requested = False
                return "reset"
            return self.frames.popleft()

    def emit(self, pcm, text):
        try:
            self.loop.call_soon_threadsafe(self.put_output, pcm)
        except RuntimeError:  # The session loop is already closed.
            self.stopped.set()
            return
        if text:
            self.session.send(Text(text=text).model_dump())
        self.spoken_frames += 1
        try:
            # One billable unit is one second of generated audio.
            self.session.add_billable_units(len(pcm) / SAMPLE_RATE)
        except RuntimeError:  # Billing is already finalized.
            self.stopped.set()

    def set_state(self, state):
        self.state = state
        self.send_status()

    def run(self):
        import torch

        torch.set_grad_enabled(False)
        try:
            with self.app.gpu_lock:
                self.converse()
        except Exception:
            logger.exception("moshi generation failed")
            self.session.send(Rejected(reason="generation failed").model_dump())
            try:
                self.loop.call_soon_threadsafe(self.end_session)
            except RuntimeError:
                pass

    def end_session(self):
        async def close_later():
            await asyncio.sleep(1)  # Let the last control message leave.
            await self.session.close()

        self.session.create_task(close_later())

    def converse(self):
        import torch

        moshi = self.app.moshi
        moshi.reset()
        skip = True
        while not self.stopped.is_set():
            frame = self.next_frame()
            if frame is None:
                return
            if isinstance(frame, str):  # A reset request starts a new conversation.
                self.set_state("restarting")
                moshi.reset()
                skip = True
                self.set_state("listening")
                continue
            chunk = torch.from_numpy(frame).to("cuda")[None, None]
            for pcm, text in moshi.step(chunk, skip=skip):
                self.emit(pcm, text)
            skip = False
            self.heard_frames += 1


class VoiceChatApp(fal.wma.App, python_version="3.12"):
    machine_type = "GPU-H100"
    requirements = [
        "fal[wma]==1.82.1",
        "moshi==0.2.13",
        "torch==2.6.0",
        "huggingface_hub>=0.24,<1",
        "av==13.1.0",
        "numpy>=1.26,<2.3",
        "pydantic>=2,<3",
        "aiortc",
    ]
    startup_timeout = 1200  # Weights download and CUDA graph capture take minutes.
    min_concurrency = 0
    max_concurrency = 2
    max_multiplexing = 1  # One session owns the GPU and the streaming state.
    keep_alive = 300
    realtime_contract = fal.wma.RealtimeContract(
        media=fal.wma.MediaContract(
            send=(fal.wma.Track(kind="audio", source="microphone", required=True),),
            receive=(fal.wma.Track(kind="audio"),),
        ),
        client_messages=Reset,
        server_messages=ServerMessage,
        client_message_examples=(
            fal.wma.MessageExample(name="New conversation", payload={"type": "reset"}),
        ),
    )

    async def setup(self):
        self.gpu_lock = threading.Lock()
        self.moshi = await asyncio.to_thread(Moshi)

    async def create_backend(self, session):
        from aiortc import RTCConfiguration

        (
            servers,
            _,
        ) = await fal.wma.RunnerIceConfig.from_bridge().build_ice_servers_async(
            session.offer.ice_servers, forwarded_status=session.offer.ice_status
        )
        voice = VoiceSession(self, session, asyncio.get_running_loop())
        session.on_message("reset", voice.on_reset)
        # Deferred cleanups run in reverse: stop the model thread, then cap billing.
        session.defer(voice.cap_billing)
        session.defer(voice.shutdown)

        def connect(pc):
            attached = False

            @pc.on("track")
            def incoming(track):
                nonlocal attached
                if track.kind == "audio" and not attached:
                    attached = True
                    speaker = make_audio_queue_track(voice.out)
                    pc.addTrack(speaker)
                    voice.start(track, speaker)

        async def expire():
            await asyncio.sleep(MAX_SESSION_SECONDS)
            await session.close()

        session.create_task(expire())
        return fal.wma.AiortcPeer(
            session,
            connect,
            rtc_configuration=RTCConfiguration(iceServers=servers),
            disconnected_grace_seconds=10,
        )
