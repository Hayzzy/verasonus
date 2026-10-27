export interface WaveformPreview {
  peaks: number[] // loudest sample per bucket, scaled to [0, 1]
  duration: number // seconds
}

/** Decode an audio file in the browser and summarise it as `count` peak values. */
export async function waveformPeaks(file: File, count = 64): Promise<WaveformPreview> {
  const context = new OfflineAudioContext(1, 1, 44_100)
  const buffer = await context.decodeAudioData(await file.arrayBuffer())
  const samples = buffer.getChannelData(0)
  const bucket = Math.max(1, Math.floor(samples.length / count))
  const peaks: number[] = []
  for (let i = 0; i < count; i++) {
    let peak = 0
    const end = Math.min(samples.length, (i + 1) * bucket)
    for (let j = i * bucket; j < end; j++) peak = Math.max(peak, Math.abs(samples[j]))
    peaks.push(peak)
  }
  const loudest = Math.max(...peaks, 1e-6)
  return { peaks: peaks.map((peak) => peak / loudest), duration: buffer.duration }
}

export function formatBytes(bytes: number): string {
  const mb = bytes / (1024 * 1024)
  return mb < 1 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${mb.toFixed(1)} MB`
}

export function formatDuration(seconds: number): string {
  const total = Math.round(seconds)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}
