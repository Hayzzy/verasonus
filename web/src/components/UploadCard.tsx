import { FileMusic, Mic, ShieldCheck, Upload, X } from 'lucide-react'
import { type DragEvent, useEffect, useId, useState } from 'react'

import { formatBytes, formatDuration, type WaveformPreview, waveformPeaks } from '../lib/audio'
import { Waveform } from './Waveform'

const ACCEPT = '.mp3,.wav,.flac,.ogg,.opus,.m4a,.aac,audio/*'
const EXTENSIONS = ['mp3', 'wav', 'flac', 'ogg', 'opus', 'm4a', 'aac']
const MAX_MB = 50

type Preview = WaveformPreview | 'loading' | 'unreadable'

/** Upload panel. Picks a file and previews it locally; analysis is wired up in Phase 3. */
export function UploadCard() {
  const inputId = useId()
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<Preview>('loading')
  const [dragging, setDragging] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

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
    setMessage(null)
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
        <span className="rounded-full bg-pill px-3 py-1 text-xs font-semibold text-apricot">
          Preview
        </span>
      </div>

      {file ? (
        <SelectedFile
          file={file}
          preview={preview}
          onClear={() => {
            setFile(null)
            setMessage(null)
          }}
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
          className={`mt-5 flex cursor-pointer flex-col items-center rounded-2xl border border-dashed px-6 py-10 text-center transition-colors focus-within:border-apricot sm:py-12 ${
            dragging ? 'border-apricot bg-card-warm' : 'border-ash hover:border-ember'
          }`}
        >
          <span className="ring-badge size-16 bg-card-warm">
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
          disabled={!file}
          onClick={() =>
            setMessage('The detector is still in training. Analysis switches on in Phase 3.')
          }
          className="h-12 rounded-full bg-linear-to-r from-flame to-ember font-display font-semibold text-white shadow-[0_10px_30px_-10px_rgb(255_122_0/0.8)] transition hover:brightness-110 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-apricot disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none disabled:hover:brightness-100"
        >
          Analyse track
        </button>
        <button
          type="button"
          disabled
          className="flex h-12 items-center justify-center gap-2 rounded-full border border-ember font-display font-semibold text-white/80 disabled:cursor-not-allowed"
        >
          <Mic className="size-4" />
          Record a clip
          <span className="rounded-full bg-pill px-2 py-0.5 text-[0.7rem] text-apricot">Phase 3</span>
        </button>
      </div>

      <p className="mt-5 flex items-start gap-2 text-sm text-mist">
        <ShieldCheck className="mt-0.5 size-4 shrink-0 text-apricot" />
        In this preview your file never leaves your device. When analysis goes live, uploads
        are deleted right after they are analysed.
      </p>
    </div>
  )
}

function SelectedFile({
  file,
  preview,
  onClear,
}: {
  file: File
  preview: Preview
  onClear: () => void
}) {
  const ready = typeof preview === 'object'
  return (
    <div className="mt-5 rounded-2xl border border-ember/70 bg-linear-to-b from-card-warm-top to-card-warm p-4 sm:p-5">
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
          className="rounded-full p-2 text-mist transition hover:bg-white/5 hover:text-white focus-visible:outline-2 focus-visible:outline-apricot"
        >
          <X className="size-5" />
        </button>
      </div>
      <div className="mt-5">
        {ready ? (
          <Waveform peaks={preview.peaks} />
        ) : (
          <p className="grid h-16 place-items-center text-sm text-mist">
            {preview === 'loading' ? 'Reading audio…' : 'This browser cannot preview this file.'}
          </p>
        )}
        <p className="mt-3 text-xs text-ash">
          Waveform of your file. Once analysis is live, the AI timeline colours it in.
        </p>
      </div>
    </div>
  )
}
