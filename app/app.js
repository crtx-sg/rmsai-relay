// rmsai companion app — per-hospital worklist (Phase 9, Step 2).
//
// Flow: PIN -> POST /session -> join the LiveKit inbox room (rmsai-inbox-<hospital_id>) -> maintain
// a live worklist from `event`/`status` data messages the orchestrator publishes. Live-push-only:
// the list is built from messages received after connect (no backlog fetch in the POC).
//
// Artifact bytes never ride the data channel — messages carry only pseudonyms + scoped links.
// Rendering artifacts inline (ECG strip / trend / report) and in-app chat come in later steps.

// Bump on every client change. Printed on load so "is the browser running the current app.js?" is
// answerable from the console instead of inferred from behaviour — a stale cached SPA looks exactly
// like a broken backend.
const APP_BUILD = "2026-08-18 ptt-2";

const LK = window.LivekitClient;

let session = null; // { url, room, token, ... } from POST /session
let room = null; // the LiveKit Room once connected
let currentSelection = null; // event_id currently scoped for chat

const ARTIFACT_LABELS = { ecg_strip: "ECG", hr_trend: "Trend", report: "Report" };
const CHAT_TOPIC = "lk.chat";

// --- worklist state: a pure reducer so the render is a function of received messages ------------
const state = { rows: new Map() }; // event_id -> row object

function applyMessage(state, msg) {
  if (!msg || !msg.event_id) return; // resilience: ignore malformed messages
  if (msg.type === "event") {
    const prev = state.rows.get(msg.event_id) || {};
    // A later `event` (e.g. replay) merges; keep an existing acknowledged status unless the
    // message carries a newer one.
    state.rows.set(msg.event_id, { ...prev, ...msg, status: msg.status || prev.status || "reported" });
  } else if (msg.type === "status") {
    const prev = state.rows.get(msg.event_id) || { event_id: msg.event_id };
    state.rows.set(msg.event_id, { ...prev, status: msg.status });
  }
}
window.applyMessage = applyMessage; // exposed for manual/console testing

// --- rendering ---------------------------------------------------------------------------------
function fmtTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return isNaN(d) ? "" : d.toLocaleTimeString();
}

function render() {
  const rows = [...state.rows.values()].sort((a, b) => (b.ts || 0) - (a.ts || 0));
  const tbody = document.getElementById("rows");
  const table = document.getElementById("table");
  const empty = document.getElementById("empty");

  empty.classList.toggle("hidden", rows.length > 0);
  table.classList.toggle("hidden", rows.length === 0);

  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    const acked = r.status === "acknowledged";
    const classes = [];
    if (acked) classes.push("acknowledged");
    if (r.event_id && r.event_id === currentSelection) classes.push("selected");
    tr.className = classes.join(" ");
    tr.dataset.eventId = r.event_id || "";
    const bed = [r.unit, r.bed].filter(Boolean).join(" / ");
    const links = r.links || {};
    // Which artifact kinds this event has (from the inbox message); the URL is minted fresh on click
    // (openArtifact), not taken from links[k].url — those publish-time links expire in ~5 min.
    const viewBtns = Object.keys(links).map((k) =>
      `<button data-view="${esc(k)}">${esc(ARTIFACT_LABELS[k] || k)}</button>`
    ).join("");
    const ackBtn = acked || !r.event_id ? ""
      : `<button data-ack="${esc(r.event_id)}">Acknowledge</button>`;
    tr.innerHTML = `
      <td>${esc(r.patient)}</td>
      <td>${esc(bed)}</td>
      <td>${esc(r.event_type)}</td>
      <td class="crit crit-${esc(r.criticality)}">${esc(r.criticality)}</td>
      <td>${esc(fmtTime(r.ts))}</td>
      <td><span class="badge ${acked ? "acknowledged" : "new"}">${esc(r.status || "new")}</span></td>
      <td>${viewBtns}${ackBtn}</td>`;
    tbody.appendChild(tr);
  }
}

// Mint a FRESH scoped link at click time, then view it. Worklist links carried in the inbox message
// expire ~5 min after publish, so we don't reuse them — we ask the gateway (POST /artifact-link, PIN
// proven by the session token) for a token that's fresh now. The chat "show" path already gets a
// fresh URL from the worker, so it calls viewArtifact directly.
async function openArtifact(kind, eventId) {
  if (!session || !eventId) return;
  try {
    const res = await fetch("/artifact-link", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ event_id: eventId, kind, session: session.token }),
    });
    if (!res.ok) return artifactError(kind, res.status);
    const { url } = await res.json();
    viewArtifact(kind, url);
  } catch (e) {
    artifactError(kind, e);
  }
}

// --- inline artifact viewer (bytes fetched from the scoped-token URL, never off the data channel) -
async function viewArtifact(kind, url) {
  const detail = document.getElementById("detail");
  detail.classList.remove("hidden");
  detail.innerHTML = `<div class="meta">Loading ${esc(kind)}…</div>`;
  try {
    if (kind === "ecg_strip") {
      detail.innerHTML = `<h2>ECG strip</h2>` +
        `<img alt="ECG strip" src="${esc(url)}" onerror="window.__artifactError('ecg_strip')" />`;
    } else if (kind === "report") {
      const res = await fetch(url);
      if (!res.ok) return artifactError(kind, res.status);
      detail.innerHTML = `<h2>Event report</h2><pre>${esc(await res.text())}</pre>`;
    } else if (kind === "hr_trend") {
      const res = await fetch(url);
      if (!res.ok) return artifactError(kind, res.status);
      const data = await res.json();
      detail.innerHTML = `<h2>HR trend</h2>` + renderSpark(data.hr_history || []);
    }
  } catch (e) {
    artifactError(kind, e);
  }
}

function renderSpark(values) {
  if (!values.length) return `<div class="meta">No HR history.</div>`;
  const w = 600, h = 120, pad = 8;
  const min = Math.min(...values), max = Math.max(...values), span = (max - min) || 1;
  const n = Math.max(values.length - 1, 1);
  const pts = values.map((v, i) => {
    const x = pad + (i * (w - 2 * pad)) / n;
    const y = h - pad - ((v - min) * (h - 2 * pad)) / span;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return `<div class="meta">HR ${min}–${max} bpm · latest ${values[values.length - 1]}</div>` +
    `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline points="${pts}"/></svg>`;
}

function artifactError(kind, info) {
  const detail = document.getElementById("detail");
  detail.classList.remove("hidden");
  const msg = info === 404 ? "link expired or unavailable" : "could not load";
  detail.innerHTML = `<div class="meta">${esc(kind)}: ${esc(msg)}</div>`;
}
window.__artifactError = (kind) => artifactError(kind, 404);

// Acknowledge an event: POST /ack (session token proves PIN). The orchestrator flips the status and
// pushes a `status` message back, but we also update optimistically so the click feels immediate.
async function ackEvent(eventId) {
  if (!session) return;
  try {
    const res = await fetch("/ack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_id: eventId, session: session.token }),
    });
    if (res.ok) {
      applyMessage(state, { type: "status", event_id: eventId, status: "acknowledged" });
      render();
    } else {
      console.warn("ack failed", res.status);
    }
  } catch (e) {
    console.warn("ack error", e);
  }
}

function esc(v) {
  return String(v == null ? "" : v).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function setStatus(text) { document.getElementById("status-pill").textContent = text; }

// --- session + connect -------------------------------------------------------------------------
async function login() {
  const pin = document.getElementById("pin").value.trim();
  const err = document.getElementById("login-err");
  err.textContent = "";
  if (!pin) { err.textContent = "Enter the PIN."; return; }
  let session;
  try {
    const res = await fetch("/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pin }),
    });
    if (!res.ok) {
      err.textContent = res.status === 401 ? "Incorrect PIN." : `Sign-in failed (${res.status}).`;
      return;
    }
    session = await res.json();
  } catch (e) {
    err.textContent = "Could not reach the gateway.";
    return;
  }
  await connect(session);
}

async function connect(sess) {
  session = sess;
  document.getElementById("login").classList.add("hidden");
  document.getElementById("worklist").classList.remove("hidden");
  // Build tag on screen, not just in the console: a stale cached page is the single most expensive
  // thing to misdiagnose here — it looks exactly like a broken worker.
  document.getElementById("room-label").textContent = `${session.room} · app ${APP_BUILD}`;
  setStatus("connecting…");

  room = new LK.Room();
  window.__room = room; // debugging

  room.on(LK.RoomEvent.DataReceived, (payload) => {
    try {
      const msg = JSON.parse(new TextDecoder().decode(payload));
      console.log("[worklist] data message:", msg);
      if (msg.type === "show") { viewArtifact(msg.kind, msg.url); return; }  // chat asked to see it
      applyMessage(state, msg);
      render();
    } catch (e) {
      console.warn("worklist: ignoring bad data message", e);
    }
  });
  // The agent's spoken (TTS) reply arrives as a remote audio track — attach it so it plays.
  room.on(LK.RoomEvent.TrackSubscribed, (track) => {
    if (track.kind === "audio") track.attach();
  });
  room.on(LK.RoomEvent.Disconnected, () => setStatus("disconnected"));
  room.on(LK.RoomEvent.Reconnecting, () => setStatus("reconnecting…"));
  room.on(LK.RoomEvent.Reconnected, () => setStatus("live"));
  // The chat worker (agent) joins a few seconds after we connect; if a row was already selected,
  // (re)send the selection so it isn't lost to the join race.
  room.on(LK.RoomEvent.ParticipantConnected, () => { if (currentSelection) sendSelect(currentSelection); });

  // The agent's typed reply arrives as a text stream on the chat topic.
  room.registerTextStreamHandler(CHAT_TOPIC, async (reader) => {
    try {
      addChatLine("assistant", await reader.readAll());
    } catch (e) {
      console.warn("chat: failed to read reply", e);
    }
  });

  try {
    await room.connect(session.url, session.token);
    setStatus("live");
    await room.startAudio().catch(() => {}); // unlock autoplay within the connect gesture
    console.log("[worklist] connected to", session.room, "as", session.identity);
  } catch (e) {
    // Surface the failure instead of leaving an empty worklist that looks like "no events".
    console.error("[worklist] connect failed", e);
    document.getElementById("worklist").classList.add("hidden");
    document.getElementById("login").classList.remove("hidden");
    document.getElementById("login-err").textContent =
      "Connected to the gateway but could not join the live room: " + ((e && e.message) || e);
  }
}

// --- per-event chat (text + push-to-talk voice), scoped by selecting a worklist row ------------
function selectEvent(eventId) {
  if (!eventId || !room) return;
  currentSelection = eventId;
  const r = state.rows.get(eventId) || {};
  document.getElementById("chat").classList.remove("hidden");
  document.getElementById("chat-title").textContent =
    `Chat — ${r.patient || "patient"} · ${r.event_type || ""}`;
  document.getElementById("chat-log").innerHTML = "";
  render(); // reflect row highlight
  sendSelect(eventId);
}

// Scope the worker's conversation to an event. Sent as a control message on the chat text channel —
// in the agents worker only lk.chat text reliably reaches the handler (raw data packets / custom
// topics are swallowed by the framework).
function sendSelect(eventId) {
  if (!room || !eventId) return;
  room.localParticipant.sendText(`/select ${eventId}`, { topic: CHAT_TOPIC })
    .catch((e) => console.warn("select failed", e));
}

function addChatLine(who, text) {
  const log = document.getElementById("chat-log");
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function sendChat() {
  const input = document.getElementById("chat-input");
  const text = input.value.trim();
  if (!text || !room) return;
  if (!currentSelection) { addChatLine("assistant", "Select an event first."); return; }
  input.value = "";
  addChatLine("clinician", text);
  try {
    await room.localParticipant.sendText(text, { topic: CHAT_TOPIC });
  } catch (e) {
    console.warn("chat: send failed", e);
    addChatLine("assistant", "Message failed to send.");
  }
}

// Push-to-talk: the mic is live only while the button is held, and the button itself delimits the
// turn. The worker runs the inbox with manual end-of-turn detection, so it needs to be *told* when
// the turn ends: releasing the button mutes the mic, which stops audio at the SFU, so the agent's
// VAD would never see the trailing silence it otherwise needs to close the turn (the turn would
// hang forever and STT would never run). Sent on the chat text channel — the one control path that
// reliably reaches the agent worker, same as /select.
const PTT_TAIL_MS = 250; // keep capturing briefly after release so the last word isn't clipped
let pttActive = false;

function sendCtl(msg) {
  if (!room) return Promise.resolve();
  return room.localParticipant.sendText(msg, { topic: CHAT_TOPIC })
    .catch((e) => console.warn("ptt: control send failed", msg, e));
}

async function pttDown() {
  console.log("[ptt] down", { hasRoom: !!room, selection: currentSelection, active: pttActive });
  if (!room || !currentSelection || pttActive) return;
  pttActive = true;
  const btn = document.getElementById("ptt");
  btn.classList.add("talking");
  btn.textContent = "🎤 Listening… release to send";
  try {
    // Open the agent's turn FIRST: it must be listening before audio flows, and this way a mic that
    // is slow to grant (or never resolves — an unplugged/busy device leaves getUserMedia pending
    // forever) can't silently swallow the control message and strand the turn.
    await sendCtl("/ptt-start");
    await room.localParticipant.setMicrophoneEnabled(true);
  } catch (e) {
    console.warn("ptt: mic enable failed", e);
    addChatLine("assistant", "Could not open the microphone — check the browser's mic permission.");
    pttActive = false;
    btn.classList.remove("talking");
    btn.textContent = "🎤 Hold to talk";
  }
}

// mouseup and mouseleave both land here; pttActive keeps the turn from being committed twice.
async function pttUp() {
  console.log("[ptt] up", { active: pttActive });
  if (!room || !pttActive) return;
  pttActive = false;
  const btn = document.getElementById("ptt");
  btn.classList.remove("talking");
  btn.textContent = "🎤 Thinking…";
  await new Promise((r) => setTimeout(r, PTT_TAIL_MS));
  try {
    await sendCtl("/ptt-end"); // agent detaches audio, flushes STT and answers
    await room.localParticipant.setMicrophoneEnabled(false);
  } finally {
    btn.textContent = "🎤 Hold to talk";
  }
}

console.log(`[app] build ${APP_BUILD}`);
document.getElementById("login-btn").addEventListener("click", login);
document.getElementById("pin").addEventListener("keydown", (e) => { if (e.key === "Enter") login(); });
// Event delegation for the per-row buttons + row selection (the table is re-rendered each message).
document.getElementById("rows").addEventListener("click", (e) => {
  const t = e.target;
  if (!t || !t.getAttribute) return;
  const ackId = t.getAttribute("data-ack");
  if (ackId) { ackEvent(ackId); return; }
  const view = t.getAttribute("data-view");
  if (view) {
    const rowEl = t.closest("tr");
    openArtifact(view, rowEl && rowEl.dataset.eventId);
    return;
  }
  // A click anywhere else on the row selects that event for chat.
  const rowEl = t.closest("tr");
  if (rowEl && rowEl.dataset.eventId) selectEvent(rowEl.dataset.eventId);
});

// Chat controls
document.getElementById("chat-send").addEventListener("click", sendChat);
document.getElementById("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendChat();
});
const pttBtn = document.getElementById("ptt");
pttBtn.addEventListener("mousedown", pttDown);
pttBtn.addEventListener("mouseup", pttUp);
pttBtn.addEventListener("mouseleave", pttUp);
pttBtn.addEventListener("touchstart", (e) => { e.preventDefault(); pttDown(); });
pttBtn.addEventListener("touchend", (e) => { e.preventDefault(); pttUp(); });
