import { defineConfig } from "vite";

export default defineConfig({
  envPrefix: ["VITE_", "FAL_"],
  server: {
    port: 5173,
    strictPort: true,
  },
});
