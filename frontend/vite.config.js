import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// Dev server proxies /api/* to the FastAPI backend so the browser only ever
// talks to one origin (no CORS preflights during development).
export default defineConfig({
  plugins: [vue()],
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
