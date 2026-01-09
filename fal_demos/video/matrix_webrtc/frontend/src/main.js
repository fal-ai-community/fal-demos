const TOKEN_EXPIRATION_SECONDS = 120;

const appIdInput = document.getElementById("appId");
const apiKeyInput = document.getElementById("apiKey");
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

const buildWsUrl = (appId, token) => {
  const normalizedAppId = normalizeAppId(appId);
  return `wss://fal.run/${normalizedAppId}?fal_jwt_token=${encodeURIComponent(token)}`;
};

const getTemporaryAuthToken = async (appId, apiKey) => {
  const { alias } = parseEndpointId(appId);
  const response = await fetch("https://rest.alpha.fal.ai/tokens/", {
    method: "POST",
    headers: {
      Authorization: `Key ${apiKey}`,
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      allowed_apps: [alias],
      token_expiration: TOKEN_EXPIRATION_SECONDS,
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
    if (event.candidate && ws) {
      ws.send(
        JSON.stringify({
          type: "icecandidate",
          candidate: {
            candidate: event.candidate.candidate,
            sdpMid: event.candidate.sdpMid,
            sdpMLineIndex: event.candidate.sdpMLineIndex,
          },
        }),
      );
    }
  };

  pc.ontrack = (event) => {
    log(`Track received: ${event.track?.kind || "unknown"}`);
    const stream = event.streams && event.streams[0]
      ? event.streams[0]
      : new MediaStream([event.track]);
    remoteVideo.srcObject = stream;
  };

  pc.addTransceiver("video", { direction: "recvonly" });
};

const sendOffer = async () => {
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  ws?.send(JSON.stringify({ type: "offer", sdp: offer.sdp }));
};

startBtn.addEventListener("click", async () => {
  if (started) return;
  started = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  logEl.textContent = "";

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

  try {
    authToken = await getTemporaryAuthToken(appId, apiKey);
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

  ws.onopen = async () => {
    setStatus("Connected");
    log("WebSocket open.");
    await ensurePeer();
    await sendOffer();
    sendActionValue("q u");
    pauseBtn.disabled = false;
  };

  ws.onmessage = async (event) => {
    let msg = null;
    try {
      msg = JSON.parse(event.data);
    } catch {
      log(`WS: ${event.data}`);
      return;
    }

    if (msg.type === "answer" && msg.sdp) {
      await pc.setRemoteDescription(new RTCSessionDescription({ type: "answer", sdp: msg.sdp }));
    } else if (msg.type === "icecandidate") {
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
    } else if (msg.type === "ready") {
      log("Server ready.");
    } else if (msg.type === "pause") {
      paused = msg.paused;
      pauseBtn.textContent = paused ? "Resume (Space)" : "Pause (Space)";
    } else {
      log(`WS message: ${event.data}`);
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
  ws.send(JSON.stringify({ type: "pause", paused }));
  pauseBtn.textContent = paused ? "Resume (Space)" : "Pause (Space)";
};

pauseBtn.addEventListener("click", togglePause);

const sendActionValue = (actionValue) => {
  if (!ws) return;
  if (paused) return;
  const payload = { type: "action", action: actionValue };
  ws.send(JSON.stringify(payload));
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
