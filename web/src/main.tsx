import '@fontsource-variable/inter'
import '@fontsource-variable/source-sans-3'
import './index.css'

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import App from './App'
import { followPointerHighlight } from './lib/glass'

followPointerHighlight()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
