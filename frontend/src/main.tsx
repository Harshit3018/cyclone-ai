import React from 'react'
import ReactDOM from 'react-dom/client'

// Self-hosted assets — bundled so the app renders correctly with no internet access.
// Previously these came from Google Fonts and unpkg via <link> tags in index.html.
import '@fontsource/inter/400.css'
import '@fontsource/inter/500.css'
import '@fontsource/inter/600.css'
import '@fontsource/inter/700.css'
import '@fontsource/inter/800.css'
import '@fontsource-variable/jetbrains-mono'
import 'leaflet/dist/leaflet.css'

import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
