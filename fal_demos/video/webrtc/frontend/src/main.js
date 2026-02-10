import { decode, encode } from "@msgpack/msgpack";

const TOKEN_EXPIRATION_SECONDS = 120;
const DEFAULT_ICE_SERVERS = [{ urls: "stun:stun.l.google.com:19302" }];

const appIdInput = document.getElementById("appId");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const remoteVideo = document.getElementById("remoteVideo");
const logEl = document.getElementById("log");
const statusEl = document.getElementById("status");
const lastKeyEl = document.getElementById("lastKey");
const clientFpsEl = document.getElementById("clientFps");

let ws = null;
let pc = null;
let started = false;
let clientFpsValue = 0;
let clientFpsLastNowMs = null;
let clientFpsCallbackId = null;
let clientFpsIntervalId = null;
let clientFpsLastVideoTime = null;

const log = (msg) => {
  const line = String(msg);
  console.log(line);
  logEl.textContent = line + "\n" + logEl.textContent;
};

const setStatus = (text) => {
  statusEl.textContent = text;
};

const setLastKey = (value) => {
  lastKeyEl.textContent = value || "none";
};

const setClientFps = (value) => {
  clientFpsEl.textContent = Number(value || 0).toFixed(1);
};

const stopClientFpsMonitor = () => {
  if (
    typeof remoteVideo.cancelVideoFrameCallback === "function" &&
    clientFpsCallbackId !== null
  ) {
    remoteVideo.cancelVideoFrameCallback(clientFpsCallbackId);
  }
  if (clientFpsIntervalId !== null) {
    window.clearInterval(clientFpsIntervalId);
  }
  clientFpsCallbackId = null;
  clientFpsIntervalId = null;
  clientFpsLastNowMs = null;
  clientFpsLastVideoTime = null;
  clientFpsValue = 0;
  setClientFps(0);
};

const startClientFpsMonitor = () => {
  stopClientFpsMonitor();

  if (typeof remoteVideo.requestVideoFrameCallback === "function") {
    const onFrame = (nowMs) => {
      if (!started || !remoteVideo.srcObject) {
        return;
      }
      if (clientFpsLastNowMs !== null) {
        const dt = nowMs - clientFpsLastNowMs;
        if (dt > 0) {
          const instantFps = 1000 / dt;
          clientFpsValue =
            clientFpsValue === 0 ? instantFps : 0.9 * clientFpsValue + 0.1 * instantFps;
          setClientFps(clientFpsValue);
        }
      }
      clientFpsLastNowMs = nowMs;
      clientFpsCallbackId = remoteVideo.requestVideoFrameCallback(onFrame);
    };
    clientFpsCallbackId = remoteVideo.requestVideoFrameCallback(onFrame);
    return;
  }

  // Fallback for older browsers: estimate from media time progression.
  clientFpsIntervalId = window.setInterval(() => {
    if (!started || !remoteVideo.srcObject) {
      return;
    }
    const t = remoteVideo.currentTime;
    if (clientFpsLastVideoTime !== null) {
      const dt = t - clientFpsLastVideoTime;
      if (dt > 0) {
        const instantFps = 1 / dt;
        clientFpsValue =
          clientFpsValue === 0 ? instantFps : 0.9 * clientFpsValue + 0.1 * instantFps;
        setClientFps(clientFpsValue);
      }
    }
    clientFpsLastVideoTime = t;
  }, 100);
};

const normalizeAppId = (value) => value.replace(/^\/+|\/+$/g, "");

const buildWsUrl = (appId, token) => {
  const normalizedAppId = normalizeAppId(appId);
  return `wss://fal.run/${normalizedAppId}?fal_jwt_token=${encodeURIComponent(token)}`;
};

const getTemporaryAuthToken = async (appId) => {
  const response = await fetch("/fal/token", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      appId,
      tokenExpirationSeconds: TOKEN_EXPIRATION_SECONDS,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Token request failed: ${response.status} ${errorBody}`);
  }

  const token = await response.json();
  if (typeof token !== "string" && token?.detail) {
    return token.detail;
  }
  return token;
};

const sendWs = (payload) => {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    return;
  }
  const encoded = encode(payload);
  const buffer = encoded.buffer.slice(
    encoded.byteOffset,
    encoded.byteOffset + encoded.byteLength,
  );
  ws.send(buffer);
};

const readWsMessage = async (data) => {
  if (data instanceof ArrayBuffer) {
    return decode(new Uint8Array(data));
  }
  if (data instanceof Blob) {
    const buffer = await data.arrayBuffer();
    return decode(new Uint8Array(buffer));
  }
  if (typeof data === "string") {
    try {
      return JSON.parse(data);
    } catch {
      return data;
    }
  }
  return data;
};

const sendAction = (keyValue) => {
  if (!ws) return;
  setLastKey(keyValue);
  sendWs({ type: "action", action: keyValue });
};

const stop = () => {
  if (ws) {
    ws.close();
    ws = null;
  }
  if (pc) {
    pc.close();
    pc = null;
  }
  remoteVideo.srcObject = null;
  stopClientFpsMonitor();
  setStatus("Disconnected");
};

const ensurePeer = async (iceServers) => {
  if (pc) return;
  pc = new RTCPeerConnection({ iceServers });

  pc.onconnectionstatechange = () => {
    log(`Peer connection state: ${pc.connectionState}`);
  };

  pc.oniceconnectionstatechange = () => {
    log(`ICE connection state: ${pc.iceConnectionState}`);
  };

  pc.onicecandidate = (event) => {
    if (!event.candidate) return;
    sendWs({
      type: "icecandidate",
      candidate: {
        candidate: event.candidate.candidate,
        sdpMid: event.candidate.sdpMid,
        sdpMLineIndex: event.candidate.sdpMLineIndex,
      },
    });
  };

  pc.ontrack = (event) => {
    const stream = event.streams && event.streams[0]
      ? event.streams[0]
      : new MediaStream([event.track]);
    remoteVideo.srcObject = stream;
    startClientFpsMonitor();
    log(`Track received: ${event.track?.kind || "unknown"}`);
  };

  pc.addTransceiver("video", { direction: "recvonly" });
};

const sendOffer = async () => {
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  sendWs({ type: "offer", sdp: offer.sdp });
};

const mapKeyboardEventToAction = (event) => {
  const key = event.key;
  if (key === " ") return "SPACE";
  if (key.length === 1) return key;
  const named = {
    ArrowUp: "UP",
    ArrowDown: "DOWN",
    ArrowLeft: "LEFT",
    ArrowRight: "RIGHT",
    Enter: "ENTER",
    Escape: "ESC",
    Tab: "TAB",
    Backspace: "BACKSPACE",
  };
  return named[key] || null;
};

window.addEventListener("keydown", (event) => {
  const target = event.target;
  if (target instanceof HTMLElement && target.closest("input, textarea, [contenteditable='true']")) {
    return;
  }
  if (!started || event.repeat) return;
  const actionValue = mapKeyboardEventToAction(event);
  if (!actionValue) return;
  sendAction(actionValue);
  event.preventDefault();
});

startBtn.addEventListener("click", async () => {
  if (started) return;
  started = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  logEl.textContent = "";
  setLastKey("none");

  const appId = normalizeAppId(appIdInput.value.trim());
  if (!appId) {
    log("Missing endpoint.");
    stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    started = false;
    return;
  }

  let authToken;
  let pendingIceServers = DEFAULT_ICE_SERVERS;
  let offerSent = false;
  try {
    authToken = await getTemporaryAuthToken(appId);
  } catch (err) {
    log(`Failed to fetch token: ${err.message || err}`);
    stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    started = false;
    return;
  }
  ws = new WebSocket(buildWsUrl(appId, authToken));
  ws.binaryType = "arraybuffer";

  ws.onopen = async () => {
    setStatus("Connected");
    log("WebSocket open.");
  };

  ws.onmessage = async (event) => {
    const msg = await readWsMessage(event.data);
    if (!msg || typeof msg !== "object") {
      log(`WS: ${String(msg)}`);
      return;
    }
    if (msg.type === "answer" && msg.sdp && pc) {
      await pc.setRemoteDescription(new RTCSessionDescription({ type: "answer", sdp: msg.sdp }));
    } else if (msg.type === "iceservers" && !offerSent) {
      if (Array.isArray(msg.iceservers) && msg.iceservers.length > 0) {
        pendingIceServers = msg.iceservers;
      }
      log(`Using ${pendingIceServers.length} ICE server entries from signaling.`);
      await ensurePeer(pendingIceServers);
      await sendOffer();
      offerSent = true;
    } else if (msg.type === "icecandidate" && pc) {
      const candidate = msg.candidate;
      if (candidate) {
        await pc.addIceCandidate(
          new RTCIceCandidate({
            candidate: candidate.candidate,
            sdpMid: candidate.sdpMid,
            sdpMLineIndex: candidate.sdpMLineIndex,
          }),
        );
      }
    } else if (msg.type === "error") {
      log(`Server error: ${msg.error}`);
    } else {
      log(`WS message: ${JSON.stringify(msg)}`);
    }
  };

  ws.onclose = (event) => {
    log(`WebSocket closed code=${event.code} reason=${event.reason}`);
    stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    started = false;
  };

  ws.onerror = (err) => {
    log("WebSocket error.");
    console.error(err);
    stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    started = false;
  };
});

stopBtn.addEventListener("click", () => {
  stop();
  startBtn.disabled = false;
  stopBtn.disabled = true;
  started = false;
});
