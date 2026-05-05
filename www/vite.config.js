import { defineConfig, searchForWorkspaceRoot } from 'vite'
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig({
  plugins: [
    wasm(),
    topLevelAwait()
  ],
  server: {
    fs:{
      allow: [
        searchForWorkspaceRoot(process.cwd()),
        "../pkg/neural_network_bg.wasm"
      ]
    }
  }
});