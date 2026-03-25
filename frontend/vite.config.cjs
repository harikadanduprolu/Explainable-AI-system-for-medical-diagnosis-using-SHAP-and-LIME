const path = require('path')
const { defineConfig } = require('vite')
const react = require('@vitejs/plugin-react')

const resolveOutDir = () => {
  const customDir = process.env.VITE_OUTPUT_DIR
  if (customDir) {
    return path.isAbsolute(customDir)
      ? customDir
      : path.resolve(__dirname, customDir)
  }
  return path.resolve(__dirname, '../backend/static')
}

module.exports = defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: resolveOutDir(),
    emptyOutDir: process.env.VITE_SKIP_EMPTY_OUTDIR === 'true' ? false : true,
  },
})
