import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Must be an absolute base. With './', index.html emits relative asset URLs
  // ("./assets/x.js"), which resolve to "/cyclone/assets/x.js" on nested routes
  // like /cyclone/:id — the SPA fallback then serves index.html for them and the
  // page renders blank.
  base: '/',
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          charts: ['recharts'],
          maps: ['leaflet'],
        },
      },
    },
  },
})
