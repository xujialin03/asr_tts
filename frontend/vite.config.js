import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import fs from "node:fs";
import path from "node:path";

// Serve everything under /public/vad/ as raw bytes. onnxruntime-web dynamically
// imports its wasm loader (.mjs); Vite's dev transform pipeline chokes on those
// (500 on `?import`), so we short-circuit them with a static file response.
function rawVadAssets() {
  const types = {
    ".mjs": "text/javascript",
    ".js": "text/javascript",
    ".wasm": "application/wasm",
    ".onnx": "application/octet-stream",
  };
  return {
    name: "raw-vad-assets",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url || "").split("?")[0];
        if (!url.startsWith("/vad/")) return next();
        const file = path.join(server.config.root, "public", url);
        if (!fs.existsSync(file)) return next();
        const ext = path.extname(file);
        res.setHeader("Content-Type", types[ext] || "application/octet-stream");
        fs.createReadStream(file).pipe(res);
      });
    },
  };
}

// Dev server proxies /api/* to the FastAPI backend so the browser only ever
// talks to one origin (no CORS preflights during development).
export default defineConfig({
  plugins: [rawVadAssets(), vue()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
