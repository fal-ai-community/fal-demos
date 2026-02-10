import { decode, encode } from "@msgpack/msgpack";

const TOKEN_EXPIRATION_SECONDS = 120;
const DEFAULT_ICE_SERVERS = [{ urls: "stun:stun.l.google.com:19302" }];

const appIdInput = document.getElementById("appId");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const pauseBtn = document.getElementById("pauseBtn");
const remoteVideo = document.getElementById("remoteVideo");
const logEl = document.getElementById("log");
const statusEl = document.getElementById("status");
const controlButtons = Array.from(document.querySelectorAll(".control-btn"));

let ws = null;
let pc = null;
let started = false;
let authToken = "";
let paused = false;

const log = (msg) => {
  console.log(msg);
  logEl.textContent = String(msg) + "\n" + logEl.textContent;
};

const setStatus = (text) => {
  statusEl.textContent = text;
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

const stop = () => {
  if (ws) {
    ws.close();
    ws = null;
  }
  if (pc) {
    pc.close();
    pc = null;
  }
  paused = false;
  pauseBtn.disabled = true;
  pauseBtn.textContent = "Pause (Space)";
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
    if (event.candidate && ws) {
      sendWs({
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
    // Helps some browsers begin playback immediately for remote tracks.
    remoteVideo.play().catch(() => {});
  };

  pc.addTransceiver("video", { direction: "recvonly" });
};

const parseIceServers = (iceServersPayload) => {
  if (!Array.isArray(iceServersPayload)) {
    return DEFAULT_ICE_SERVERS;
  }
  const servers = iceServersPayload.filter((item) => item && item.urls);
  return servers.length > 0 ? servers : DEFAULT_ICE_SERVERS;
};

const sendOffer = async () => {
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  sendWs({ type: "offer", sdp: offer.sdp });
};

const sendWs = (payload) => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
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

startBtn.addEventListener("click", async () => {
  if (started) return;
  started = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  logEl.textContent = "";

  const appId = normalizeAppId(appIdInput.value.trim());
  if (!appId) {
    log("Missing endpoint.");
    stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    started = false;
    return;
  }

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

  const wsUrl = buildWsUrl(appId, authToken);
  ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  ws.onopen = async () => {
    setStatus("Connected");
    log("WebSocket open.");
    pauseBtn.disabled = false;
  };

  let pendingIceServers = DEFAULT_ICE_SERVERS;
  let offerSent = false;
  let negotiatingOffer = false;

  const ensureOfferSent = async () => {
    if (offerSent || negotiatingOffer) return;
    negotiatingOffer = true;
    offerSent = true;
    try {
      await ensurePeer(pendingIceServers);
      await sendOffer();
      sendActionValue("q u");
      log("Sent offer. Waiting for answer...");
    } catch (err) {
      offerSent = false;
      throw err;
    } finally {
      negotiatingOffer = false;
    }
  };

  ws.onmessage = async (event) => {
    const msg = await readWsMessage(event.data);
    if (!msg || typeof msg !== "object") {
      log(`WS: ${String(msg)}`);
      return;
    }

    if (msg.type === "iceservers" && !offerSent) {
      pendingIceServers = parseIceServers(msg.iceservers);
      log(`Using ${pendingIceServers.length} ICE server entries from signaling.`);
      await ensureOfferSent();
    } else if (msg.type === "answer" && msg.sdp && pc) {
      await pc.setRemoteDescription(new RTCSessionDescription({ type: "answer", sdp: msg.sdp }));
    } else if (msg.type === "icecandidate" && pc) {
      const c = msg.candidate;
      if (c) {
        await pc.addIceCandidate(new RTCIceCandidate({
          candidate: c.candidate,
          sdpMid: c.sdpMid,
          sdpMLineIndex: c.sdpMLineIndex,
        }));
      }
    } else if (msg.type === "error") {
      log(`Server error: ${msg.error}`);
    } else if (msg.type === "stream_exhausted") {
      log("Stream exhausted.");
    } else if (msg.type === "pause") {
      paused = msg.paused;
      pauseBtn.textContent = paused ? "Resume (Space)" : "Pause (Space)";
    } else {
      log(`WS message: ${JSON.stringify(msg)}`);
    }
  };

  ws.onclose = (ev) => {
    log(`WebSocket closed code=${ev.code} reason=${ev.reason}`);
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

const togglePause = () => {
  if (!ws) return;
  paused = !paused;
  sendWs({ type: "pause", paused });
  pauseBtn.textContent = paused ? "Resume (Space)" : "Pause (Space)";
};

pauseBtn.addEventListener("click", togglePause);

const sendActionValue = (actionValue) => {
  if (!ws) return;
  if (paused) return;
  const payload = { type: "action", action: actionValue };
  sendWs(payload);
};

const sendMove = (move) => {
  sendActionValue(`${move} u`);
};

const sendLook = (look) => {
  sendActionValue(`q ${look}`);
};

const handleKeydown = (event) => {
  const target = event.target;
  if (target instanceof HTMLElement && target.closest("input, textarea, [contenteditable='true']")) {
    return;
  }
  if (event.repeat) return;
  const key = event.key.toLowerCase();
  if (key === " ") {
    togglePause();
    event.preventDefault();
    return;
  }
  if (["w", "a", "s", "d"].includes(key)) {
    sendMove(key);
    event.preventDefault();
  } else if (["i", "j", "k", "l"].includes(key)) {
    sendLook(key);
    event.preventDefault();
  }
};

window.addEventListener("keydown", handleKeydown);

controlButtons.forEach((button) => {
  const move = button.dataset.move;
  const look = button.dataset.look;
  const send = () => {
    if (move) {
      sendMove(move);
    } else if (look) {
      sendLook(look);
    }
  };
  button.addEventListener("pointerdown", () => {
    button.classList.add("active");
    send();
  });
  button.addEventListener("pointerup", () => {
    button.classList.remove("active");
  });
  button.addEventListener("pointerleave", () => {
    button.classList.remove("active");
  });
});
