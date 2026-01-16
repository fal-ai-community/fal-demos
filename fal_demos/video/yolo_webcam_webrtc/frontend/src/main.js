import { createFalClient } from "@fal-ai/client";

const appIdInput = document.getElementById("appId");
const apiKeyInput = document.getElementById("apiKey");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const localVideo = document.getElementById("localVideo");
const remoteVideo = document.getElementById("remoteVideo");
const localFpsEl = document.getElementById("localFps");
const remoteFpsEl = document.getElementById("remoteFps");
const localResEl = document.getElementById("localRes");
const remoteResEl = document.getElementById("remoteRes");
const localBitrateEl = document.getElementById("localBitrate");
const remoteBitrateEl = document.getElementById("remoteBitrate");
const logEl = document.getElementById("log");
const statusEl = document.getElementById("status");

let connection = null;
let pc = null;
let started = false;
let localStream = null;
let localFpsStop = null;
let remoteFpsStop = null;
let bitrateStop = null;
let falClient = null;
let serverReady = false;
let localStreamReady = false;
let offerSent = false;

const envApiKey = import.meta.env.FAL_KEY || import.meta.env.VITE_FAL_KEY;
if (envApiKey) {
  apiKeyInput.value = envApiKey;
}

const log = (msg) => {
  console.log(msg);
  logEl.textContent = String(msg) + "\n" + logEl.textContent;
};

const setStatus = (text) => {
  statusEl.textContent = text;
};

const normalizeAppId = (value) => value.replace(/^\/+|\/+$/g, "");

const ensureEndpointIdFormat = (id) => {
  const parts = id.split("/");
  if (parts.length >= 3) {
    return id;
  }
  throw new Error(`Invalid endpoint: ${id}. Use <appOwner>/<appId>/webrtc.`);
};

const parseEndpointId = (id) => {
  const normalizedId = ensureEndpointIdFormat(id);
  const parts = normalizedId.split("/");
  return {
    owner: parts[0],
    alias: parts[1],
    path: parts.slice(2).join("/") || undefined,
  };
};

const encodeJsonMessage = (input) => {
  if (input instanceof Uint8Array) {
    return input;
  }
  const payload = typeof input === "string" ? input : JSON.stringify(input);
  return new TextEncoder().encode(payload);
};

const decodeJsonMessage = async (data) => {
  if (typeof data === "string") {
    return JSON.parse(data);
  }
  if (data instanceof ArrayBuffer || data instanceof Uint8Array) {
    return JSON.parse(new TextDecoder().decode(data));
  }
  if (data instanceof Blob) {
    return JSON.parse(
      new TextDecoder().decode(new Uint8Array(await data.arrayBuffer())),
    );
  }
  return data;
};

const stop = () => {
  if (connection) {
    connection.close();
    connection = null;
  }
  if (pc) {
    pc.close();
    pc = null;
  }
  if (localStream) {
    localStream.getTracks().forEach((track) => track.stop());
    localStream = null;
  }
  localVideo.srcObject = null;
  remoteVideo.srcObject = null;
  if (localFpsStop) {
    localFpsStop();
    localFpsStop = null;
  }
  if (remoteFpsStop) {
    remoteFpsStop();
    remoteFpsStop = null;
  }
  if (bitrateStop) {
    bitrateStop();
    bitrateStop = null;
  }
  localFpsEl.textContent = "FPS: --";
  remoteFpsEl.textContent = "FPS: --";
  localResEl.textContent = "-- x --";
  remoteResEl.textContent = "-- x --";
  localBitrateEl.textContent = "Up: -- Mbps";
  remoteBitrateEl.textContent = "Down: -- Mbps";
  setStatus("Disconnected");
  serverReady = false;
  localStreamReady = false;
  offerSent = false;
};

const ensurePeer = async () => {
  if (pc) return;
  pc = new RTCPeerConnection({ iceServers: [{ urls: "stun:stun.l.google.com:19302" }] });

  pc.onconnectionstatechange = () => {
    log(`Peer connection state: ${pc.connectionState}`);
  };

  pc.oniceconnectionstatechange = () => {
    log(`ICE connection state: ${pc.iceConnectionState}`);
  };

  pc.onicecandidate = (event) => {
    if (event.candidate && connection) {
      log("Sending icecandidate");
      connection.send({
        type: "icecandidate",
        candidate: {
          candidate: event.candidate.candidate,
          sdpMid: event.candidate.sdpMid,
          sdpMLineIndex: event.candidate.sdpMLineIndex,
        },
      });
    }
  };

  pc.ontrack = (event) => {
    log(`Track received: ${event.track?.kind || "unknown"}`);
    const stream = event.streams && event.streams[0]
      ? event.streams[0]
      : new MediaStream([event.track]);
    remoteVideo.srcObject = stream;
    if (remoteFpsStop) {
      remoteFpsStop();
    }
    remoteFpsStop = startFpsCounter(remoteVideo, remoteFpsEl, remoteResEl);
  };
};

const attachLocalStream = async () => {
  if (localStream) return;
  localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  localVideo.srcObject = localStream;
  localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));
  localStreamReady = true;
  maybeSendOffer();
  if (localFpsStop) {
    localFpsStop();
  }
  localFpsStop = startFpsCounter(localVideo, localFpsEl, localResEl);
  if (!bitrateStop) {
    bitrateStop = startBitrateMonitor();
  }
};

const maybeSendOffer = () => {
  if (!offerSent && serverReady && pc && localStreamReady) {
    offerSent = true;
    sendOffer();
  }
};

const startFpsCounter = (video, outputEl, resEl) => {
  let frameCount = 0;
  let lastTime = performance.now();
  let rafId = null;
  let stop = false;
  let lastRes = "";

  const update = (timestamp) => {
    if (stop) return;
    frameCount += 1;
    const delta = timestamp - lastTime;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (width && height) {
      const nextRes = `${width} x ${height}`;
      if (nextRes !== lastRes) {
        resEl.textContent = nextRes;
        lastRes = nextRes;
      }
    }
    if (delta >= 1000) {
      const fps = Math.round((frameCount * 1000) / delta);
      outputEl.textContent = `FPS: ${fps}`;
      frameCount = 0;
      lastTime = timestamp;
    }
    if (video.requestVideoFrameCallback) {
      video.requestVideoFrameCallback((_, metadata) => update(metadata.expectedDisplayTime || performance.now()));
    } else {
      rafId = requestAnimationFrame(update);
    }
  };

  if (video.requestVideoFrameCallback) {
    video.requestVideoFrameCallback((_, metadata) => update(metadata.expectedDisplayTime || performance.now()));
  } else {
    rafId = requestAnimationFrame(update);
  }

  return () => {
    stop = true;
    if (rafId) {
      cancelAnimationFrame(rafId);
    }
  };
};

const startBitrateMonitor = () => {
  let lastOutBytes = null;
  let lastOutTime = null;
  let lastInBytes = null;
  let lastInTime = null;

  const update = async () => {
    if (!pc) return;
    const stats = await pc.getStats();
    stats.forEach((report) => {
      if (report.type === "outbound-rtp" && report.kind === "video") {
        if (lastOutBytes !== null && lastOutTime !== null) {
          const bytes = report.bytesSent;
          const time = report.timestamp;
          const bitrate = ((bytes - lastOutBytes) * 8) / ((time - lastOutTime) / 1000);
          localBitrateEl.textContent = `Up: ${Math.max(0, bitrate / 1_000_000).toFixed(2)} Mbps`;
        }
        lastOutBytes = report.bytesSent;
        lastOutTime = report.timestamp;
      }
      if (report.type === "inbound-rtp" && report.kind === "video") {
        if (lastInBytes !== null && lastInTime !== null) {
          const bytes = report.bytesReceived;
          const time = report.timestamp;
          const bitrate = ((bytes - lastInBytes) * 8) / ((time - lastInTime) / 1000);
          remoteBitrateEl.textContent = `Down: ${Math.max(0, bitrate / 1_000_000).toFixed(2)} Mbps`;
        }
        lastInBytes = report.bytesReceived;
        lastInTime = report.timestamp;
      }
    });
  };

  const intervalId = setInterval(() => {
    update().catch((err) => log(`Bitrate stats error: ${err.message || err}`));
  }, 1000);

  return () => clearInterval(intervalId);
};

const sendOffer = async () => {
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  log("Sending offer");
  connection?.send({ type: "offer", sdp: offer.sdp });
};

startBtn.addEventListener("click", async () => {
  if (started) return;
  started = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  logEl.textContent = "";

  if (!window.__falWsPatched) {
    const OriginalWebSocket = window.WebSocket;
    class LoggingWebSocket extends OriginalWebSocket {
      constructor(url, protocols) {
        log(`WS url: ${url}`);
        super(url, protocols);
        this.addEventListener("open", () => log("WS open"));
        this.addEventListener("close", (ev) =>
          log(`WS close code=${ev.code} reason=${ev.reason || "n/a"}`),
        );
        this.addEventListener("error", () => log("WS error"));
      }
    }
    // Preserve readyState constants without mutating read-only properties.
    ["CONNECTING", "OPEN", "CLOSING", "CLOSED"].forEach((key) => {
      if (key in OriginalWebSocket) {
        try {
          Object.defineProperty(LoggingWebSocket, key, {
            value: OriginalWebSocket[key],
            writable: false,
            enumerable: true,
          });
        } catch {
          /* best-effort */
        }
      }
    });
    window.WebSocket = LoggingWebSocket;
    window.__falWsPatched = true;
  }

  const appId = normalizeAppId(appIdInput.value.trim());
  const apiKey = apiKeyInput.value.trim();
  if (!appId || !apiKey) {
    log("Missing endpoint or API key.");
    stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    started = false;
    return;
  }

  let endpoint = "";
  let path = "";
  try {
    const parsed = parseEndpointId(appId);
    endpoint = `${parsed.owner}/${parsed.alias}`;
    path = parsed.path ?? "";
  } catch (err) {
    log(err.message || err);
    stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    started = false;
    return;
  }
  const connectionKey = `webrtc:${endpoint}:${path || "realtime"}`;
  log(`App ID: ${appId}`);
  log(`Endpoint: ${endpoint}`);
  log(`Path: ${path || "(default /realtime)"}`);
  log(`Connection key: ${connectionKey}`);

  falClient = createFalClient({
    credentials: apiKey,
    suppressLocalCredentialsWarning: true,
  });

  connection = falClient.realtime.connect(endpoint, {
    connectionKey,
    path: path || undefined,
    throttleInterval: 0,
    encodeMessage: encodeJsonMessage,
    decodeMessage: decodeJsonMessage,
    onResult: async (msg) => {
      if (msg.type === "answer" && msg.sdp) {
        await pc.setRemoteDescription(
          new RTCSessionDescription({ type: "answer", sdp: msg.sdp }),
        );
      } else if (msg.type === "icecandidate") {
        const c = msg.candidate;
        if (c) {
          await pc.addIceCandidate(
            new RTCIceCandidate({
              candidate: c.candidate,
              sdpMid: c.sdpMid,
              sdpMLineIndex: c.sdpMLineIndex,
            }),
          );
        }
      } else if (msg.type === "error") {
        log(`Server error: ${msg.error}`);
      } else if (msg.type === "ready") {
        setStatus("Connected");
        log("Server ready.");
        serverReady = true;
        maybeSendOffer();
      } else {
        log(`WS message: ${JSON.stringify(msg)}`);
      }
    },
    onError: (err) => {
      log(`Realtime error: ${err.message || err}`);
      stop();
      startBtn.disabled = false;
      stopBtn.disabled = true;
      started = false;
    },
  });
  log("Triggering connection");
  connection.send({ type: "hello" });

  setStatus("Connecting");
  try {
    await ensurePeer();
    await attachLocalStream();
  } catch (err) {
    log(`Failed to start: ${err.message || err}`);
    stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    started = false;
  }
});

stopBtn.addEventListener("click", () => {
  stop();
  startBtn.disabled = false;
  stopBtn.disabled = true;
  started = false;
});
