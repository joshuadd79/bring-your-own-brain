/**
 * bringYourOwnBrain.org: Execution Engine (100% Hugging Face weights, jsDelivr for libraries; no GitHub at runtime)
 *
 * ONE model family, TWO backends, automatic fallback:
 *   GPU  -> WebLLM (MLC build):  Qwen2.5-0.5B-Instruct-q4f16_1-MLC   (weights from huggingface.co)
 *   CPU  -> Transformers.js:     onnx-community/Qwen2.5-0.5B-Instruct, q4 WASM   (weights from huggingface.co)
 * If the GPU path can't start for ANY reason (no WebGPU, no shader-f16, missing model library, error), the CPU path is used.
 */

const TJS_SOURCES = [
  'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1',
  'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/+esm'
];
const WEBLLM_URL = 'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.82/+esm';   // jsDelivr build of @mlc-ai/web-llm

const MODEL_NAME = 'Qwen2.5 0.5B Instruct';
const GPU_MODEL = {
  id: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  // WebLLM's compiled GPU library is hosted on GitHub by default. We self-host it on bringyourownbrain.org instead
  // (see the setup notes). Save the file with exactly this name in the /models/ folder next to this script.
};
const CPU_MODEL = { id: "onnx-community/Qwen2.5-0.5B-Instruct", name: "Qwen 2.5 0.5B" };

const isExtension = typeof chrome !== 'undefined' && !!(chrome.runtime && chrome.runtime.id);
const WASM_THREADS = 1; // 1 thread keeps the browser event loop responsive

let tjs = null;
let webllm = null;
let sharedCaches = null;  // <-- ADD THIS LINE!
export function setSharedCaches(c) { 
  sharedCaches = c; 
  if (c) {
    try { Object.defineProperty(globalThis, 'caches', { value: c, configurable: true, writable: true }); } catch {}
    try { Object.defineProperty(window, 'caches', { value: c, configurable: true, writable: true }); } catch {}
  }
}

let webllmEngine = null;   // GPU backend
let modelGen = null;       // CPU backend
let activeLoadedTier = null;
let gpuAvailable = false;
let activeDevice = null;   // 'webgpu' | 'wasm'

// ---- interrupt plumbing (one generation runs at a time, so these flags are per-job) ----
const CANCEL_MSG = '__BYOB_CANCELLED__';
let cpuCancel = false;        // CPU path: checked before every token
let generating = false;       // GPU path: only interrupt while a generation is really running
let currentStopping = null;   // CPU path: Transformers.js InterruptableStoppingCriteria, when available
function interruptEngine() {
  cpuCancel = true;
  try { if (currentStopping) currentStopping.interrupt(); } catch {}
  try { if (generating && webllmEngine && typeof webllmEngine.interruptGenerate === 'function') webllmEngine.interruptGenerate(); } catch {}
}

// GPU capability check
const gpuReady = (async () => {
  try {
    gpuAvailable = !!(navigator.gpu && await navigator.gpu.requestAdapter());
  } catch {
    gpuAvailable = false;
  }
})();

async function getTjs() {
  if (tjs) return tjs;
  let lastErr = null;
  for (const src of TJS_SOURCES) {
    try {
      tjs = await import(new URL(src, location.href).href);
      if (src.includes('vendor')) {
        tjs.env.backends.onnx.wasm.wasmPaths = new URL('./vendor/', location.href).href;
      }
      break;
    } catch (e) {
      lastErr = e;
    }
  }
  if (!tjs) throw new Error('Could not load Transformers.js (' + (lastErr?.message || lastErr) + ')');
  tjs.env.allowLocalModels = false;
  tjs.env.useBrowserCache = true;
  tjs.env.backends.onnx.wasm.numThreads = WASM_THREADS;
  return tjs;
}

// Lazy, so a missing/offline copy can never break the CPU path.
const getWebllm = async () => webllm || (webllm = await import(WEBLLM_URL));

async function applyCache() {
  if (sharedCaches) {
    // Route Transformers.js model files through the UNPARTITIONED Cache API (StorageAccessHandle.caches).
    const c = await sharedCaches.open('byob-models');
    tjs.env.useBrowserCache = false;
    tjs.env.useCustomCache = true;
    tjs.env.customCache = { match: (k) => c.match(k), put: (k, r) => c.put(k, r) };
  }
}

async function unloadActiveEngine() {
  if (webllmEngine) {
    try { await webllmEngine.unload(); } catch {}
    webllmEngine = null;
  }
  if (modelGen) {
    try { await modelGen.dispose(); } catch {}
    modelGen = null;
  }
  activeLoadedTier = null;
  activeDevice = null;
}

// ---- CPU backend (Transformers.js, WASM) ----
async function loadCpu(onProgress) {
  await getTjs();
  await applyCache();
  activeDevice = 'wasm';
  modelGen = await tjs.pipeline('text-generation', CPU_MODEL.id, {
    device: 'wasm',
    dtype: CPU_MODEL.dtype,
    progress_callback: onProgress
  });

  // Yield to the event loop per token so the tab never freezes
  if (modelGen?.model && typeof modelGen.model.forward === 'function') {
    const origForward = modelGen.model.forward;
    modelGen.model.forward = async function (...args) {
      await new Promise((r) => setTimeout(r, 0));
      if (cpuCancel) throw new Error(CANCEL_MSG);   // stop within one token
      return origForward.apply(this, args);
    };
  }
  return modelGen;
}

// ---- GPU backend (WebLLM, MLC build) ----
// ---- GPU backend (WebLLM, MLC build) ----
async function loadGpu(onProgress) {
  const w = await getWebllm();
  activeDevice = 'webgpu';
  webllmEngine = await w.CreateMLCEngine(GPU_MODEL.id, {
    initProgressCallback: (r) => {
      if (onProgress) onProgress({ status: 'pct', pct: Math.round((r.progress || 0) * 100), text: r.text || '' });
    }
  });
}

// GPU first, CPU as the automatic fallback
async function loadTier(tier = 'micro', onProgress) {
  if (activeLoadedTier === tier && (webllmEngine || modelGen)) return;

  await gpuReady;
  await unloadActiveEngine();

  if (gpuAvailable) {
    try {
      await loadGpu(onProgress);
      activeLoadedTier = tier;
      return;
    } catch (e) {
      console.warn('WebGPU engine failed, falling back to CPU (WASM):', e);
      await unloadActiveEngine();
      if (onProgress) onProgress({ status: 'pct', pct: 1, text: 'GPU not available, switching to CPU…' });
    }
  }

  await loadCpu(onProgress);
  activeLoadedTier = tier;
}

// Transport
const replyTargets = new Map();

function emit(streamId, type, text) {
  if (isExtension) {
    try {
      const p = chrome.runtime.sendMessage({ target: 'content_script', streamId, type, text });
      if (p && p.catch) p.catch(() => {});
    } catch {}
    return;
  }
  const t = replyTargets.get(streamId);
  const dest = t?.source || window;
  const origin = t?.source && t.source !== window && t.origin && t.origin !== 'null' ? t.origin : '*';
  try {
    dest.postMessage({ type: 'BYOB_STREAM_EVENT', streamId, eventType: type, text }, origin);
  } catch {}
  if (type === 'DONE' || type === 'ERROR' || type === 'CANCELLED') replyTargets.delete(streamId);
}

async function getActiveTier() {
  if (isExtension) {
    const data = await chrome.storage.local.get(['activeTier']);
    return data.activeTier || 'micro';
  }
  try { return localStorage.getItem('byob_activeTier') || 'micro'; } catch { return 'micro'; }
}

// ---------------------------------------------------------------------------
// Priority scheduler. There is ONE model on ONE device, so only one generation runs at a time.
// But a more important request can preempt a less important one:
//   high   = the player is waiting     (always goes first, preempts everything below it)
//   normal = quick reactions           (preempts background chatter)
//   low    = background chatter / pre-cooked lines (first to be cut off)
// Cancelled or dropped requests get a CANCELLED stream event. Nothing is ever left hanging.
export const PRIORITY = { low: 1, normal: 2, high: 3 };
const QUEUE_CAP = { 1: 2, 2: 4, 3: 12 };   // per priority: oldest waiting request is dropped first

export function createScheduler({ run, emit: emitFn, interrupt }) {
  const queue = [];
  let running = null;
  let seq = 0;
  const prioOf = (p) => (typeof p === 'number' ? Math.min(3, Math.max(1, Math.round(p))) : PRIORITY[p] || PRIORITY.normal);

  function dropQueued(job, reason) { emitFn(job.id, 'CANCELLED', reason); job.done(); }

  function trim(prio) {
    const cap = QUEUE_CAP[prio] ?? 4;
    const same = queue.filter((j) => j.prio === prio);
    while (same.length > cap) { const old = same.shift(); queue.splice(queue.indexOf(old), 1); dropQueued(old, 'dropped'); }
  }

  function pump() {
    if (running || !queue.length) return;
    queue.sort((a, b) => b.prio - a.prio || a.seq - b.seq);
    const job = queue.shift();
    running = job;
    Promise.resolve().then(() => run(job)).catch(() => {}).finally(() => { running = null; job.done(); pump(); });
  }

  function submit(id, payload = {}) {
    const job = { id, payload, prio: prioOf(payload && payload.priority), seq: ++seq, cancelReason: null };
    const p = new Promise((res) => { job.done = res; });
    queue.push(job);
    trim(job.prio);
    if (running && running.prio < job.prio && !running.cancelReason) {
      running.cancelReason = 'preempted';
      try { interrupt(running); } catch {}
    }
    pump();
    return p;
  }

  function cancel(id, reason = 'cancelled') {
    if (running && running.id === id) {
      if (!running.cancelReason) { running.cancelReason = reason; try { interrupt(running); } catch {} }
      return true;
    }
    const i = queue.findIndex((j) => j.id === id);
    if (i >= 0) { const [job] = queue.splice(i, 1); dropQueued(job, reason); return true; }
    return false;
  }

  return { submit, cancel, state: () => ({ running: running && running.id, queued: queue.map((j) => j.id) }) };
}

const scheduler = createScheduler({ run: runChat, emit, interrupt: interruptEngine });

// Router
if (isExtension) {
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.target !== 'offscreen') return;

    if (message.action === 'SWAP_ENGINE') {
      loadTier(message.tier)
        .then(() => sendResponse({ success: true, tier: message.tier }))
        .catch((err) => sendResponse({ success: false, error: err.message }));
      return true;
    }

    if (message.action === 'UNINSTALL_STORAGE') {
      unloadActiveEngine().then(() => sendResponse({ success: true }));
      return true;
    }

    if (message.action === 'CHAT_COMPLETION') {
      scheduler.submit(message.streamId, message.payload)
        .then((res) => sendResponse({ success: true, res }))
        .catch((err) => sendResponse({ success: false, error: err.message }));
      return true;
    }

    if (message.action === 'CANCEL_CHAT') {
      sendResponse({ success: scheduler.cancel(message.streamId, 'cancelled') });
      return true;
    }
  });
} else {
  window.addEventListener('message', (event) => {
    const d = event.data;
    if (!d || typeof d !== 'object' || typeof d.type !== 'string' || !d.type.startsWith('BYOB_')) return;
    const src = event.source || window;
    const origin = event.origin && event.origin !== 'null' && src !== window ? event.origin : '*';
    const say = (msg) => { try { src.postMessage(msg, origin); } catch {} };

    const gate = globalThis.BYOB_GATE;
    const allowed = typeof gate === 'function' ? !!gate(event) : true;

    if (d.type === 'BYOB_PING') {
      getActiveTier().then((tier) => say({
        type: 'BYOB_PONG', id: d.id, extensionInstalled: true, brand: 'bringYourOwnBrain.org',
        activeTier: tier, installedTiers: [tier], isAuthorized: allowed
      }));
    } else if (d.type === 'BYOB_REQUEST_AUTH') {
      say({ type: 'BYOB_AUTH_RESULT', id: d.id, success: allowed });
    } else if (d.type === 'BYOB_CHAT') {
      if (!allowed) {
        say({ type: 'BYOB_STREAM_EVENT', streamId: d.id, eventType: 'ERROR', text: 'Not connected. Finish the bringYourOwnBrain connect steps first.' });
        return;
      }
      if (typeof d.id !== 'string' || !d.id) return;
      replyTargets.set(d.id, { source: src, origin: event.origin });
      scheduler.submit(d.id, d.payload || {}).catch(() => {});
    } else if (d.type === 'BYOB_CANCEL') {
      if (!allowed || typeof d.id !== 'string') return;
      const t = replyTargets.get(d.id);
      if (t && t.origin === event.origin) scheduler.cancel(d.id, 'cancelled');   // only the page that asked can cancel it
    }
  });
  window.postMessage({ type: 'BYOB_READY', brand: 'bringYourOwnBrain.org', version: '1.1.0', timestamp: Date.now() }, '*');
}

async function runChat(job) {
  const streamId = job.id;
  const payload = job.payload || {};
  const cancelled = () => !!job.cancelReason;
  cpuCancel = false;
  generating = false;
  currentStopping = null;
  try {
    if (cancelled()) { emit(streamId, 'CANCELLED', job.cancelReason); return; }

    const tier = await getActiveTier();
    await loadTier(tier);
    if (!webllmEngine && !modelGen) throw new Error('Model is not loaded.');
    if (cancelled()) { emit(streamId, 'CANCELLED', job.cancelReason); return; }

    // Standard chat roles: system / user / assistant
    const messages = (payload.messages || []).map((m) => {
      let role = m.role || 'user';
      if (role === 'developer') role = 'system';
      else if (role === 'model') role = 'assistant';
      else if (role !== 'system' && role !== 'user' && role !== 'assistant') role = 'user';
      return { role, content: m.content || '' };
    });

    const maxTokens = Math.min(Math.max(payload.max_tokens || 120, 1), 1024);
    const temperature = typeof payload.temperature === 'number' ? payload.temperature : 0.7;

    if (webllmEngine) {
      // GPU (WebLLM)
      generating = true;
      const stream = await webllmEngine.chat.completions.create({
        messages, stream: true, max_tokens: maxTokens, temperature
      });
      for await (const chunk of stream) {
        if (cancelled()) continue;          // let the interrupted stream finish on its own
        const text = chunk.choices?.[0]?.delta?.content || '';
        if (text) emit(streamId, 'CHUNK', text);
      }
      generating = false;
    } else {
      // CPU (Transformers.js)
      const streamer = new tjs.TextStreamer(modelGen.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: payload.skip_special_tokens !== false,
        callback_function: (text) => { if (text && !cancelled()) emit(streamId, 'CHUNK', text); }
      });

      let promptInput = messages;
      try {
        if (modelGen.tokenizer && typeof modelGen.tokenizer.apply_chat_template === 'function') {
          promptInput = modelGen.tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });
        }
      } catch {
        promptInput = messages
          .map((m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>`)
          .join('\n') + '\n<|im_start|>assistant\n';
      }

      const opts = {
        max_new_tokens: maxTokens,
        streamer,
        return_full_text: false,
        do_sample: temperature > 0,
        temperature: temperature > 0 ? temperature : undefined
      };
      if (typeof tjs.InterruptableStoppingCriteria === 'function') {
        currentStopping = new tjs.InterruptableStoppingCriteria();
        opts.stopping_criteria = currentStopping;
      }
      try {
        await modelGen(promptInput, opts);
      } catch (e) {
        if (!(e && e.message === CANCEL_MSG)) throw e;   // our own cancel is not an error
      }
      currentStopping = null;
    }

    if (cancelled()) emit(streamId, 'CANCELLED', job.cancelReason);
    else emit(streamId, 'DONE');
  } catch (err) {
    generating = false;
    currentStopping = null;
    if (cancelled() || (err && err.message === CANCEL_MSG)) emit(streamId, 'CANCELLED', job.cancelReason || 'cancelled');
    else emit(streamId, 'ERROR', (err && err.message) || String(err));
  }
}

export function getEngineInfo() {
  return { tier: activeLoadedTier, device: activeDevice, gpu: gpuAvailable, name: MODEL_NAME };
}
export { loadTier, unloadActiveEngine };
