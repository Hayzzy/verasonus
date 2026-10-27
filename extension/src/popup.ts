// Phase 4 skeleton: popup script. Will start a capture and render the verdict.

const version = document.querySelector<HTMLParagraphElement>('#version')
if (version) version.textContent = `v${chrome.runtime.getManifest().version} · skeleton`
