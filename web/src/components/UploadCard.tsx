import { FileMusic, LoaderCircle, Mic, ShieldCheck, Upload, X } from 'lucide-react'
import { type DragEvent, useEffect, useId, useRef, useState } from 'react'

import { type AnalysisResult, analyzeTrack } from '../lib/api'
import { formatBytes, formatDuration, type WaveformPreview, waveformPeaks } from '../lib/audio'
import { ResultPanel } from './ResultPanel'
import { Waveform } from './Waveform'

const ACCEPT = '.mp3,.wav,.flac,.ogg,.opus,.m4a,.aac,audio/*'
const EXTENSIONS = ['mp3', 'wav', 'flac', 'ogg', 'opus', 'm4a', 'aac']
const MAX_MB = 50

type Preview = WaveformPreview | 'loading' | 'unreadable'

/** Upload panel: pick a file, preview it locally, send it to the API and show the result. */
export function UploadCard() {
  const inputId = useId()
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<Preview>('loading')
  const [dragging, setDragging] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [analysing, setAnalysing] = useState(false)
  const request = useRef<AbortController | null>(null)

  async function analyse() {
    if (!file) return
    request.current?.abort()
    const controller = new AbortController()
    request.current = controller
    setAnalysing(true)
    setMessage(null)
    try {
      setResult(await analyzeTrack(file, controller.signal))
    } catch (error) {
      if (!controller.signal.aborted) setMessage((error as Error).message)
    } finally {
      if (request.current === controller) setAnalysing(false)
    }
  }

  function clear() {
    request.current?.abort()
    setFile(null)
    setResult(null)
    setAnalysing(false)
    setMessage(null)
  }

  useEffect(() => {
    if (!file) return
    let cancelled = false
    setPreview('loading')
    waveformPeaks(file).then(
      (result) => !cancelled && setPreview(result),
      () => !cancelled && setPreview('unreadable'),
    )
    return () => {
      cancelled = true
    }
  }, [file])

  function choose(candidate: File | undefined) {
    if (!candidate) return
    const extension = candidate.name.split('.').pop()?.toLowerCase() ?? ''
    if (!candidate.type.startsWith('audio/') && !EXTENSIONS.includes(extension)) {
      setMessage('That file does not look like audio. Try MP3, WAV, FLAC, OGG or M4A.')
      return
    }
    if (candidate.size > MAX_MB * 1024 * 1024) {
      setMessage(`Files up to ${MAX_MB} MB for now.`)
      return
    }
    clear()
    setFile(candidate)
  }

  function onDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault()
    setDragging(false)
    choose(event.dataTransfer.files[0])
  }

  function onDragLeave(event: DragEvent<HTMLElement>) {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false)
  }

  return (
    <div className="card relative bg-card/85 p-5 shadow-[0_40px_120px_-40px_rgb(255_122_0/0.55)] backdrop-blur-xl sm:p-7">
      <div className="flex items-center justify-between gap-4">
        <p className="eyebrow">Analyse a track</p>
        <span className="glass rounded-full px-3 py-1 text-xs font-semibold text-apricot">
          Prototype
        </span>
      </div>

      {file ? (
        <SelectedFile
          file={file}
          preview={preview}
          result={result}
          onClear={clear}
        />
      ) : (
        <label
          htmlFor={inputId}
          onDragOver={(event) => {
            event.preventDefault()
            setDragging(true)
          }}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          className={`glass-tile mt-5 flex cursor-pointer flex-col items-center rounded-2xl border border-dashed px-6 py-10 text-center focus-within:border-apricot sm:py-12 ${
            dragging ? 'border-apricot bg-card-warm' : 'border-ash hover:border-ember'
          }`}
        >
          <span className="ring-badge glass-tile size-16">
            <Upload className="size-7" strokeWidth={1.5} />
          </span>
          <span className="mt-5 font-display text-xl font-semibold tracking-[-0.02em]">
            Drop a song or clip here
          </span>
          <span className="mt-1.5 text-mist">
            or <span className="text-apricot underline underline-offset-4">browse your files</span>
          </span>
          <span className="mt-4 text-sm text-ash">MP3, WAV, FLAC, OGG or M4A · up to {MAX_MB} MB</span>
          <input
            id={inputId}
            type="file"
            accept={ACCEPT}
            className="sr-only"
            onChange={(event) => {
              choose(event.target.files?.[0])
              event.target.value = ''
            }}
          />
        </label>
      )}

      {message && (
        <p role="status" className="mt-4 text-[0.95rem] text-apricot">
          {message}
        </p>
      )}

      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        <button
          type="button"
          disabled={!file || analysing}
          onClick={analyse}
          className="glass-primary h-12 rounded-full font-display font-semibold text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-apricot disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
        >
          {analysing ? (
            <span className="flex items-center justify-center gap-2">
              <LoaderCircle className="size-4 animate-spin" /> Analysing…
            </span>
          ) : result ? (
            'Analyse again'
          ) : (
            'Analyse track'
          )}
        </button>
        <button
          type="button"
          disabled
          className="glass flex h-12 items-center justify-center gap-2 rounded-full font-display font-semibold text-white/80 disabled:cursor-not-allowed"
        >
          <Mic className="size-4" />
          Record a clip
          <span className="rounded-full bg-pill px-2 py-0.5 text-[0.7rem] text-apricot">Soon</span>
        </button>
      </div>

      <p className="mt-5 flex items-start gap-2 text-sm text-mist">
        <ShieldCheck className="mt-0.5 size-4 shrink-0 text-apricot" />
        Your file is sent to the VeraSonus API only when you press Analyse, and deleted as soon
        as it has been analysed.
      </p>
    </div>
  )
}

function SelectedFile({
  file,
  preview,
  result,
  onClear,
}: {
  file: File
  preview: Preview
  result: AnalysisResult | null
  onClear: () => void
}) {
  const ready = typeof preview === 'object'
  return (
    <div className="glass-tile mt-5 rounded-2xl border border-ember/70 p-4 sm:p-5">
      <div className="flex items-center gap-4">
        <span className="ring-badge size-12 shrink-0">
          <FileMusic className="size-5" strokeWidth={1.6} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate font-display font-semibold tracking-[-0.01em]">{file.name}</p>
          <p className="text-sm text-mist">
            {formatBytes(file.size)}
            {ready && ` · ${formatDuration(preview.duration)}`}
          </p>
        </div>
        <button
          type="button"
          onClick={onClear}
          aria-label="Remove file"
          className="glass rounded-full p-2 text-mist hover:text-white focus-visible:outline-2 focus-visible:outline-apricot"
        >
          <X className="size-5" />
        </button>
      </div>
      <div className="mt-5">
        {ready ? (
          <Waveform
            peaks={preview.peaks}
            segments={result?.segments}
            duration={result?.duration_seconds ?? preview.duration}
          />
        ) : (
          <p className="grid h-16 place-items-center text-sm text-mist">
            {preview === 'loading' ? 'Reading audio…' : 'This browser cannot preview this file.'}
          </p>
        )}
        <p className="mt-3 text-xs text-ash">
          {result
            ? 'Coloured by AI probability: orange sounds AI-generated, grey sounds human.'
            : 'Waveform of your file. After analysis, the AI timeline colours it in.'}
        </p>
      </div>
      {result && <ResultPanel result={result} />}
    </div>
  )
}
