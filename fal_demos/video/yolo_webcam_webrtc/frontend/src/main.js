const TOKEN_EXPIRATION_SECONDS = 120;

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

let ws = null;
let pc = null;
let started = false;
let authToken = "";
let localStream = null;
let localFpsStop = null;
let remoteFpsStop = null;
let bitrateStop = null;

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
  if (localFpsStop) {
    localFpsStop();
  }
  localFpsStop = startFpsCounter(localVideo, localFpsEl, localResEl);
  if (!bitrateStop) {
    bitrateStop = startBitrateMonitor();
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
    try {
      await attachLocalStream();
    } catch (err) {
      log(`Failed to get webcam: ${err.message || err}`);
      stop();
      startBtn.disabled = false;
      stopBtn.disabled = true;
      started = false;
      return;
    }
    await sendOffer();
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
    } else if (msg.type === "ready") {
      log("Server ready.");
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
