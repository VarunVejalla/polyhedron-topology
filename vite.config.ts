import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const repoBase = "/polyhedron-topology/";

// eslint-disable-next-line import/no-unused-modules
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: mode === "production" ? repoBase : "/",
  server: {
    port: 5173,
    strictPort: false,
  },
}));
