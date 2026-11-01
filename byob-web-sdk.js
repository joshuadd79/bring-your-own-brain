/**
 * bringYourOwnBrain.org: Web SDK for games (no extension needed). CLASSIC SCRIPT, works from any domain, no CORS setup.
 *
 *   <script src="https://bringyourownbrain.org/byob-web-sdk.js"></script>
 *   <script>
 *     const brain = new BringYourOwnBrainWeb({ onState: (s) => console.log(s) });
 *     brain.mount();                     // puts the connect button (an iframe of the hub) on the page
 *     const reply = await brain.chat([{ role: 'user', content: 'Hello' }]);
 *     const p = brain.chat(msgs, { priority: 'low', onChunk: t => ui(t) });  p.cancel();  // background request you can stop
 *   </script>
 *
 * The hub (connect.html) is found next to this script, so the SDK and hub always come from the same place.
 */
(function () {
'use strict';
if (window.BringYourOwnBrainWeb) return;
const SELF = document.currentScript && document.currentScript.src;
const DEFAULT_HUB = SELF ? new URL('connect.html', SELF).href : 'https://bringyourownbrain.org/connect.html';
class BringYourOwnBrainWeb {
  constructor(options = {}) {
      this.ask = this.chat.bind(this);
  this.generate = this.chat.bind(this);
  this.prompt = this.chat.bind(this);
    this.hubUrl = options.hubUrl || DEFAULT_HUB;
    this.hubOrigin = new URL(this.hubUrl, location.href).origin;
    this.container = options.container || null;           // null = floating bottom-right
    this.onChunk = options.onChunk || null;
    this.onState = options.onState || null;
    this.compact = { width: options.width || '240px', height: options.height || '52px' };
    this.state = 'idle';
    this.iframe = null;
    this._waiters = [];
    this._pending = new Set();
    this.version = '0.3.0';
    this._onMessage = this._onMessage.bind(this);
  }

  mount() {
    if (this.iframe) return this.iframe;
    const f = document.createElement('iframe');
    f.src = this.hubUrl;
    f.title = 'bringYourOwnBrain local AI';
    f.setAttribute('allow', 'storage-access');
    f.style.cssText = 'border:0;background:transparent;';
    this.iframe = f;
    this._layout(false);
    (this.container || document.body).appendChild(f);
    window.addEventListener('message', this._onMessage);
    // Handshake: tell the hub who we are. Retry until it answers (covers slow loads and reloads).
    let tries = 0;
    const hello = () => {
      if (this._hubSeen || tries++ > 40 || !this.iframe) return;
      try { f.contentWindow.postMessage({ type: 'BYOB_HELLO' }, this.hubOrigin); } catch {}
      setTimeout(hello, 500);
    };
    f.addEventListener('load', () => { this._hubSeen = false; tries = 0; hello(); });
    return f;
  }

  /**
   * Ask the hub to open its connect flow right now. Use it after YOUR OWN "Connect" button was clicked,
   * so the player does not have to click a second button. Nothing is loaded from the hub until you mount() or open().
   */
  open() {
    if (!this.iframe) this.mount();
    try { this.iframe.contentWindow.postMessage({ type: 'BYOB_OPEN' }, this.hubOrigin); } catch {}
  }

  _layout(expanded) {
    const s = this.iframe.style;
    if (expanded) {
      Object.assign(s, { position: 'fixed', inset: '0', width: '100vw', height: '100vh', zIndex: '2147483647', borderRadius: '0' });
    } else if (this.container) {
      Object.assign(s, { position: 'static', inset: 'auto', width: this.compact.width, height: this.compact.height, zIndex: 'auto', borderRadius: '12px' });
    } else {
      Object.assign(s, { position: 'fixed', inset: 'auto 16px 16px auto', width: this.compact.width, height: this.compact.height, zIndex: '2147483000', borderRadius: '12px' });
    }
  }

  _setState(state, message) {
    this.state = state;
    if (this.onState) this.onState(state, message);
    if (state === 'ready') { this._waiters.forEach((w) => w.resolve({ success: true })); this._waiters = []; }
  }

  _onMessage(e) {
    if (!this.iframe || e.source !== this.iframe.contentWindow || e.origin !== this.hubOrigin) return;
    const d = e.data || {};
    this._hubSeen = true;
    if (d.type === 'BYOB_UI') this._layout(d.action === 'expand');
    else if (d.type === 'BYOB_STATE') this._setState(d.state, d.message);
  }

  /** Resolves when the player has completed the connect flow and the model is ready. */
  connect() {
    if (!this.iframe) this.mount();
    if (this.state === 'ready') return Promise.resolve({ success: true });
    return new Promise((resolve, reject) => this._waiters.push({ resolve, reject }));
  }

  /**
   * Ask the local model. Returns a Promise<string> (the full reply) with two extras attached:
   *   p.id        the request id
   *   p.cancel()  stop this request (queued or running). The promise then rejects with err.cancelled === true
   *               (err.reason: 'cancelled' | 'preempted' | 'dropped', err.partial: text received so far).
   *
   * options:
   *   priority     'high' (a person is waiting) | 'normal' (default) | 'low' (background chatter)
   *                A higher priority request preempts a lower one that is running. Low ones are dropped first.
   *   onChunk(t)   streamed text for THIS request (falls back to the constructor's onChunk)
   *   max_tokens, temperature, ...  passed to the model
   */
  chat(messages, options = {}) {
       if (typeof messages === 'string') {
      messages = [{ role: 'user', content: messages }];
    }
    const { onChunk: perRequestChunk, ...rest } = options || {};   // functions can't cross postMessage
    const id = 'byob_req_' + Date.now() + '_' + Math.random().toString(36).slice(2, 7);
    let cancelNow = () => {};
    const p = new Promise((resolve, reject) => {
      if (this.state !== 'ready' || !this.iframe) { reject(new Error('bringYourOwnBrain is not connected yet.')); return; }
      const frame = this.iframe;
      let text = '';
      let settled = false;
      const entry = { abort: (msg) => finish(reject, new Error(msg)) };
      const finish = (fn, v) => {
        if (settled) return;
        settled = true;
        window.removeEventListener('message', handler);
        this._pending.delete(entry);
        fn(v);
      };
      const handler = (e) => {
        if (e.source !== frame.contentWindow || e.origin !== this.hubOrigin) return;
        const d = e.data;
        if (!d || d.type !== 'BYOB_STREAM_EVENT' || d.streamId !== id) return;
        if (d.eventType === 'CHUNK') {
          text += d.text;
          const cb = perRequestChunk || this.onChunk;
          if (cb) { try { cb(d.text); } catch {} }
        } else if (d.eventType === 'DONE') finish(resolve, text);
        else if (d.eventType === 'ERROR') finish(reject, new Error(d.text || 'Inference error'));
        else if (d.eventType === 'CANCELLED') {
          const err = new Error('cancelled');
          err.cancelled = true; err.reason = d.text || 'cancelled'; err.partial = text;
          finish(reject, err);
        }
      };
      window.addEventListener('message', handler);
      this._pending.add(entry);
      cancelNow = () => {
        if (settled) return;
        try { frame.contentWindow.postMessage({ type: 'BYOB_CANCEL', id }, this.hubOrigin); } catch {}
      };
      try { frame.contentWindow.postMessage({ type: 'BYOB_CHAT', id, payload: { messages, ...rest } }, this.hubOrigin); }
      catch (err) { finish(reject, err); }
    });
    p.id = id;
    p.cancel = () => cancelNow();
    return p;
  }

  destroy() {
    for (const p of [...this._pending]) p.abort('bringYourOwnBrain was disconnected.');
    window.removeEventListener('message', this._onMessage);
    if (this.iframe) this.iframe.remove();
    this.iframe = null;
  }
}

window.BringYourOwnBrainWeb = BringYourOwnBrainWeb;
})();
