// Phase 4 skeleton: background service worker.
//
// Planned flow when the user clicks "Analyse this tab" in the popup:
//   1. chrome.tabCapture.getMediaStreamId({ targetTabId }) for the active tab
//   2. chrome.offscreen.createDocument({ url: 'offscreen.html', reasons: ['USER_MEDIA'], ... })
//   3. the offscreen document records 10-20 s of tab audio and POSTs it to the API (/analyze)
//   4. the verdict comes back here and is shown in the popup

chrome.runtime.onInstalled.addListener(() => {
  console.info('VeraSonus extension installed (Phase 4 skeleton, capture not active yet).')
})
