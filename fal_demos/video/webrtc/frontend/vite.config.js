import { defineConfig } from "vite";

const DEFAULT_TOKEN_EXPIRATION_SECONDS = 120;

const extractAlias = (appId = "") => {
  const normalized = String(appId).trim().replace(/^\/+|\/+$/g, "");
  const parts = normalized.split("/").filter(Boolean);
  if (parts.length < 2) {
    throw new Error(`Invalid endpoint "${appId}". Use <owner>/<app>/realtime.`);
  }
  if (parts[parts.length - 1] === "webrtc" || parts[parts.length - 1] === "realtime") {
    parts.pop();
  }
  return parts[1];
};

const readBody = (req) =>
  new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
    });
    req.on("end", () => resolve(data));
    req.on("error", reject);
  });

const falTokenMiddleware = () => ({
  name: "fal-token-middleware",
  configureServer(server) {
    server.middlewares.use("/fal/token", async (req, res) => {
      if (req.method !== "POST") {
        res.statusCode = 405;
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ error: "Method not allowed" }));
        return;
      }

      const apiKey = process.env.FAL_KEY;
      if (!apiKey) {
        res.statusCode = 500;
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ error: "Missing FAL_KEY on the dev server." }));
        return;
      }

      try {
        const rawBody = await readBody(req);
        const body = rawBody ? JSON.parse(rawBody) : {};
        const alias = extractAlias(body.appId);
        const tokenExpirationSeconds =
          Number(body.tokenExpirationSeconds) || DEFAULT_TOKEN_EXPIRATION_SECONDS;

        const response = await fetch("https://rest.alpha.fal.ai/tokens/", {
          method: "POST",
          headers: {
            Authorization: `Key ${apiKey}`,
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({
            allowed_apps: [alias],
            token_expiration: tokenExpirationSeconds,
          }),
        });

        const payload = await response.text();
        res.statusCode = response.status;
        res.setHeader("Content-Type", "application/json");
        res.end(payload);
      } catch (error) {
        res.statusCode = 500;
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ error: error.message || String(error) }));
      }
    });

    server.middlewares.use("/fal/ice", async (req, res) => {
      if (req.method !== "GET") {
        res.statusCode = 405;
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ error: "Method not allowed" }));
        return;
      }

      const credentialsUrl = process.env.METERED_TURN_CREDENTIALS_URL;
      const meteredApiKey = process.env.METERED_TURN_API_KEY;
      if (!credentialsUrl || !meteredApiKey) {
        res.statusCode = 500;
        res.setHeader("Content-Type", "application/json");
        res.end(
          JSON.stringify({
            error: "Missing METERED_TURN_CREDENTIALS_URL or METERED_TURN_API_KEY on the dev server.",
          }),
        );
        return;
      }

      try {
        const query = new URLSearchParams({ apiKey: meteredApiKey }).toString();
        const joinChar = credentialsUrl.includes("?") ? "&" : "?";
        const url = `${credentialsUrl}${joinChar}${query}`;
        const response = await fetch(url, {
          method: "GET",
          headers: { Accept: "application/json" },
        });
        const payload = await response.text();
        res.statusCode = response.status;
        res.setHeader("Content-Type", "application/json");
        res.end(payload);
      } catch (error) {
        res.statusCode = 500;
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ error: error.message || String(error) }));
      }
    });
  },
});

export default defineConfig({
  envPrefix: ["VITE_", "FAL_"],
  plugins: [falTokenMiddleware()],
  server: {
    port: 5173,
    strictPort: true,
  },
});
