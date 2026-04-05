import path from 'path'
import { fileURLToPath } from 'url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const resolveOutDir = () => {
  const customDir = process.env.VITE_OUTPUT_DIR
  if (customDir) {
    return path.isAbsolute(customDir)
      ? customDir
      : path.resolve(__dirname, customDir)
  }
  return path.resolve(__dirname, '../backend/static')
}

const apiTarget = process.env.VITE_API_TARGET || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: resolveOutDir(),
    emptyOutDir: process.env.VITE_SKIP_EMPTY_OUTDIR === 'true' ? false : true,
  },
})
