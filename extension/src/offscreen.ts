// Phase 4 skeleton: offscreen document. Service workers cannot use getUserMedia or
// MediaRecorder, so recording will happen here:
//   navigator.mediaDevices.getUserMedia({
//     audio: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: streamId } },
//   })
//   -> MediaRecorder for 10-20 s -> Blob -> fetch(`${API_URL}/analyze`, { method: 'POST', body })
// Keep the tab audible by routing the captured stream to an AudioContext destination.
// Tab audio is already lossy-compressed by streaming services, so codec robustness matters.

export {}
