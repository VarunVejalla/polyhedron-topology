import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// eslint-disable-next-line import/no-unused-modules
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
  },
});
