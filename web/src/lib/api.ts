// Mirrors src/verasonus/api/schemas.py. Keep the two in sync.

export type Verdict = 'likely_ai' | 'partly_ai' | 'uncertain' | 'likely_human'

export interface Segment {
  start_seconds: number
  end_seconds: number
  ai_probability: number
}

export interface StemScore {
  name: 'vocals' | 'instrumental'
  ai_probability: number
}

export interface GeneratorGuess {
  name: string // "unknown" when the track looks unlike every known generator
  probability: number
}

export interface FeatureContribution {
  name: string
  contribution: number // positive pushes towards "AI"
}

export interface AnalysisResult {
  ai_probability: number
  verdict: Verdict
  confidence: number
  duration_seconds: number
  segments: Segment[]
  stems: StemScore[] // separated vocals and instrumental, scored on their own
  generator: GeneratorGuess | null
  explanation: { summary: string; top_features: FeatureContribution[] }
  model_version: string
}

// In development Vite forwards /api to FastAPI on port 8000 (see vite.config.ts).
const API_URL = import.meta.env.VITE_API_URL || '/api'

/** Send a track to the API for analysis. */
export async function analyzeTrack(file: File, signal?: AbortSignal): Promise<AnalysisResult> {
  const body = new FormData()
  body.append('file', file)
  const response = await fetch(`${API_URL}/analyze`, { method: 'POST', body, signal })
  if (!response.ok) {
    const detail = await response.json().then((b) => b?.detail, () => null)
    throw new Error(typeof detail === 'string' ? detail : `Analysis failed (${response.status})`)
  }
  return (await response.json()) as AnalysisResult
}
