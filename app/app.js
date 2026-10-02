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
const APP_BUILD = "2026-10-02 trend-policy-1";

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

// Model confidence in the *rhythm*. An event can reach this worklist on the vitals override alone
// (deteriorating patient, uncertain classification), so a bare event type would present a coin-flip
// read as an asserted finding. Flagged when the relay marked it low-confidence.
function fmtConfidence(r) {
  if (typeof r.confidence !== "number") return ""; // older publisher / status-only row
  const pct = `${Math.round(r.confidence * 100)}%`;
  return r.low_confidence
    ? `<span class="conf low" title="Below the low-confidence threshold — the rhythm is uncertain">⚠ ${pct}</span>`
    : `<span class="conf">${pct}</span>`;
}

// What the alert rests on. A vitals-driven row leads with the vital that triggered it and demotes
// the rhythm to "unconfirmed" — the relay could not stand behind that classification, so the row
// must not print it as a finding.
function fmtEvent(r) {
  const rhythm = esc(r.event_type);
  if (r.alert_basis !== "vitals") return rhythm;
  const why = r.alert_reason ? esc(r.alert_reason) : "vitals escalation";
  // A normal rhythm isn't "unconfirmed" — it's confirmed benign, and simply not the reason we called.
  const note = r.event_type === "NORMAL_SINUS" ? "rhythm reads normal" : `unconfirmed: ${rhythm}`;
  return `<span class="vitals-led">Vitals alert</span><span class="sub">${why}</span>` +
    `<span class="sub">${note}</span>`;
}

// Data-source badge (demo data only: the relay never sends one for a real device) and, when the
// event has a ground truth, the outcome against it.
function fmtBadges(r) {
  const src = r.source ? `<span class="src" title="data source">${esc(r.source)}</span>` : "";
  const oc = r.outcome
    ? `<span class="oc oc-${esc(r.outcome)}" title="truth: ${esc(r.truth)}">${esc(fmtOutcome(r.outcome))}</span>`
    : "";
  return src || oc ? `<span class="sub">${src}${oc}</span>` : "";
}

function fmtOutcome(code) {
  return ({ TP: "TP ✓", TP_WRONG_CLASS: "TP · wrong class", FP: "FP", FN: "FN", TN: "TN",
            UNSCORABLE: "unscorable" })[code] || code || "";
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
      <td>${fmtEvent(r)}${fmtBadges(r)}${r.why ? `<span class="why" title="${esc(r.why)}">${esc(r.why)}</span>` : ""}</td>
      <td>${fmtConfidence(r)}</td>
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
  document.getElementById("tabs").classList.remove("hidden");
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
      if (msg.type === "event") schedulePerfRefresh(); // live: the perf tab follows new events
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
  showEventInfo(eventId, "detail");
}

// --- why this event / where it came from (POST /event-info, session-gated) ------------------------
async function showEventInfo(eventId, targetId) {
  if (!session || !eventId) return;
  const el = document.getElementById(targetId);
  el.classList.remove("hidden");
  el.innerHTML = `<div class="meta">Loading…</div>`;
  try {
    const res = await fetch("/event-info", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ event_id: eventId, session: session.token }),
    });
    if (!res.ok) { el.innerHTML = `<div class="meta">No details (${res.status}).</div>`; return; }
    el.innerHTML = renderInfo(await res.json());
  } catch (e) {
    el.innerHTML = `<div class="meta">Could not load details.</div>`;
  }
}

// One vital trend: how far it moved over how long, judged against the hospital's policy (minimum
// change + normal range), expandable to the readings it was computed from with a mini chart. The
// Mann-Kendall p-value is only a tooltip: it measures how *consistent* the drift is, not its size.
const TREND_REASON = {
  below_min_change: (d) => `below the ${d.min_change} ${d.unit} threshold`,
  within_normal: (d) => `within normal ${d.normal[0]}–${d.normal[1]}`,
  toward_normal: (d) => `returning toward normal ${d.normal[0]}–${d.normal[1]}`,
  away_from_normal: (d) => `outside normal ${d.normal[0]}–${d.normal[1]}, ≥ ${d.min_change} ${d.unit}`,
};

function fmtSpan(s) {
  const m = Math.round(s / 60);
  return m < 120 ? `${m} min` : `${(m / 60).toFixed(1)} h`;
}

function renderTrend(d) {
  const vals = (d.samples || []).map((s) => s.v);
  let head, tip = "";
  if (d.change != null) {
    const sign = d.change > 0 ? "+" : "";
    const why = TREND_REASON[d.reason] ? ` — ${TREND_REASON[d.reason](d)}` : "";
    head = `${esc(d.vital)} ${esc(d.direction)} ${sign}${esc(d.change)} ${esc(d.unit)}` +
      `${d.span_s ? ` over ${fmtSpan(d.span_s)}` : ""}${d.latest != null ? ` (now ${esc(d.latest)})` : ""}${esc(why)}`;
    if (d.p != null) {
      tip = ` title="Mann-Kendall p=${esc(Number(d.p).toPrecision(2))}: how unlikely a drift this consistent is by chance. It measures consistency, not size; the threshold and normal range decide whether it matters."`;
    }
  } else {  // stored before the clinical policy: statistical verdict only
    const p = d.p != null ? ` (p=${esc(d.p < 0.001 ? "<0.001" : Number(d.p).toPrecision(2))})` : "";
    head = `${esc(d.vital)} ${esc(d.direction)}${p}`;
  }
  if (!vals.length) return `<div${tip}>${head}</div>`;
  // the span is already in the head when the policy fields are present
  const span = d.change == null && d.samples.length > 1
    ? Math.round((d.samples[d.samples.length - 1].t - d.samples[0].t) / 60) : 0;
  const n = `${vals.length} samples${span ? ` over ${span} min` : ""}`;
  return `<details><summary${tip}>${head} · ${n}</summary>` +
    `<div class="samples">${miniSpark(vals)} ${vals.map(esc).join(" → ")}</div></details>`;
}

function miniSpark(values) {
  if (values.length < 2) return "";
  const w = 200, h = 34, min = Math.min(...values), max = Math.max(...values), rng = (max - min) || 1;
  const pts = values.map((v, i) => `${((i * w) / (values.length - 1)).toFixed(1)},${(h - 2 - ((v - min) * (h - 4)) / rng).toFixed(1)}`).join(" ");
  return `<svg class="mini" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline points="${pts}"/></svg>`;
}

// --- theme: black or white background; remembered per browser (best-effort) --------------------
function currentTheme() {
  const set = document.documentElement.dataset.theme;
  if (set) return set;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("rmsai-theme", next); } catch (e) { /* storage unavailable: session only */ }
  updateThemeButton();
}

function updateThemeButton() {
  const b = document.getElementById("theme-btn");
  if (b) b.textContent = currentTheme() === "dark" ? "☀ Light" : "☾ Dark";
}

function pct(x) { return typeof x === "number" ? `${Math.round(x * 100)}%` : "–"; }

function renderInfo(i) {
  const x = i.explanation || {};
  const rh = x.rhythm || {}, vit = x.vitals || {}, crit = x.criticality || {}, dec = x.decision || {};
  const mews = vit.mews || {};
  const comps = (mews.components || []).map((c) => `${esc(c.name)} ${esc(c.value)} → ${esc(c.score)}`);
  const trends = (vit.deteriorating || []).map(renderTrend);
  const quiet = (vit.not_alerting || []).map(renderTrend);
  const prov = i.provenance || {};
  const provRows = Object.entries(prov).map(([k, v]) => `<dt>${esc(k.replace(/_/g, " "))}</dt><dd>${esc(v)}</dd>`).join("");
  const delivery = Object.entries(i.delivery || {}).map(([k, v]) => `${esc(k)}: ${esc(v)}`).join(" · ") || "–";
  const truth = i.truth
    ? `<dt>truth</dt><dd>${esc(i.truth)} <span class="oc oc-${esc(i.outcome)}">${esc(fmtOutcome(i.outcome))}</span></dd>`
    : `<dt>truth</dt><dd>— (no ground truth${i.unscorable_reason ? `: ${esc(i.unscorable_reason)}` : ""})</dd>`;
  return `<div class="info">
    <h2>${esc(i.patient)} · ${esc(i.predicted)}</h2>
    <div class="headline">${esc(i.why || "")}</div>
    <h3>Rhythm</h3><dl>
      <dt>model says</dt><dd>${esc(rh.event_type || i.predicted)} at ${pct(rh.confidence ?? i.confidence)}
        (${esc(rh.status || "")}; threshold ${pct(rh.threshold)})</dd>${truth}
      <dt>model</dt><dd>${esc(i.model_id || "–")}</dd></dl>
    <h3>Vitals</h3><dl>
      <dt>MEWS</dt><dd>${esc(mews.score)} (${esc(mews.risk)}; threshold ${esc(mews.threshold)})${comps.length ? ` — ${comps.join(", ")}` : ""}</dd>
      <dt>deteriorating</dt><dd>${trends.length ? trends.join("") : "none"}</dd>${quiet.length
        ? `<dt>trends, not alerting</dt><dd>${quiet.join("")}</dd>` : ""}</dl>
    <h3>Criticality &amp; decision</h3><dl>
      <dt>criticality</dt><dd>${esc(crit.level || i.criticality)}${crit.escalated_by && crit.escalated_by.length
        ? ` (from ${esc(crit.base)}, raised by ${crit.escalated_by.map(esc).join(", ")})` : ""}</dd>
      <dt>alert</dt><dd>${dec.dispatch ? "yes" : "no"} — ${esc(dec.summary || dec.reason_code || "")}</dd>
      <dt>delivered</dt><dd>${delivery}</dd></dl>
    <h3>Data source</h3><dl><dt>source</dt><dd>${esc(i.source)}</dd>${provRows}</dl>
  </div>`;
}

// --- model performance tab (POST /metrics) ---------------------------------------------------------
let perfTimer = null;

function showTab(tab) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.getElementById("wl-view").classList.toggle("hidden", tab !== "worklist");
  document.getElementById("perf-view").classList.toggle("hidden", tab !== "perf");
  if (tab === "perf") loadPerf();
}

function perfVisible() { return !document.getElementById("perf-view").classList.contains("hidden"); }

function schedulePerfRefresh() {
  if (!perfVisible()) return;
  clearTimeout(perfTimer);
  perfTimer = setTimeout(loadPerf, 1500); // a burst of events → one refresh
}

async function loadPerf() {
  if (!session) return;
  const status = document.getElementById("perf-status");
  status.textContent = "loading…";
  const body = {
    session: session.token,
    since: document.getElementById("f-since").value || null,
    dataset: document.getElementById("f-dataset").value || null,
    model: document.getElementById("f-model").value || null,
  };
  try {
    const res = await fetch("/metrics", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
    });
    if (!res.ok) { status.textContent = `could not load (${res.status})`; return; }
    renderPerf(await res.json());
    status.textContent = `updated ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    status.textContent = "could not reach the gateway";
  }
}

function fillSelect(id, values) {
  const sel = document.getElementById(id);
  const current = sel.value;
  const known = new Set([...sel.options].map((o) => o.value));
  for (const v of values) if (v && !known.has(v)) sel.add(new Option(v, v));
  sel.value = current;
}

function tile(label, m) {
  if (!m) return `<div class="tile"><div class="k">${esc(label)}</div><div class="v">–</div><div class="ci">no data</div></div>`;
  return `<div class="tile"><div class="k">${esc(label)}</div><div class="v">${pct(m.value)}</div>` +
    `<div class="ci">${m.k}/${m.n} · 95% CI ${pct(m.low)}–${pct(m.high)}</div></div>`;
}

function renderConfusion(cm) {
  const truths = Object.keys(cm);
  const preds = [...new Set(truths.flatMap((t) => Object.keys(cm[t])))].sort();
  if (!truths.length) return "";
  const short = (c) => esc(c.replace("VENTRICULAR_", "V-").replace("ATRIAL_", "A-").replace(/_/g, " "));
  let h = `<table class="cm"><tr><th>truth ↓ / pred →</th>${preds.map((p) => `<th>${short(p)}</th>`).join("")}</tr>`;
  for (const t of truths) {
    h += `<tr><th>${short(t)}</th>` + preds.map((p) => {
      const n = cm[t][p] || 0;
      const cls = n ? (p === t ? "diag" : "off") : "";
      return `<td class="${cls}">${n || ""}</td>`;
    }).join("") + `</tr>`;
  }
  return h + `</table>`;
}

function renderPerClass(pc) {
  const rows = Object.entries(pc);
  if (!rows.length) return "";
  return `<table class="pc"><tr><th>class</th><th>support</th><th>precision</th><th>recall</th><th>F1</th></tr>` +
    rows.map(([c, v]) => `<tr><td style="text-align:left">${esc(c)}</td><td>${v.support}</td>` +
      `<td>${v.precision ? pct(v.precision.value) : "–"}</td><td>${v.recall ? pct(v.recall.value) : "–"}</td>` +
      `<td>${v.f1 ?? "–"}</td></tr>`).join("") + `</table>`;
}

function renderPerf(data) {
  fillSelect("f-dataset", [...new Set(data.events.map((e) => e.dataset).filter(Boolean))]);
  fillSelect("f-model", data.models.map((m) => m.model_id));
  const blocks = data.models.map((m) => {
    const s = m.summary, c = s.counts, a = s.alert;
    const small = s.scored < 30
      ? `<div class="small-n">Small sample (${s.scored} scored): intervals are wide — a demo, not an evaluation.</div>` : "";
    return `<div class="model-block"><h2>${esc(m.model_id)}</h2>
      <div class="meta">${s.scored} scored of ${s.events} labelled events · sources: ${m.sources.map(esc).join(", ")}${s.unscorable ? ` · ${s.unscorable} unscorable` : ""}</div>
      ${small}
      <div class="meta">TP ${c.TP} (wrong class ${c.TP_wrong_class}) · FP ${c.FP} · FN ${c.FN} · TN ${c.TN}</div>
      <div class="tiles">${tile("exact class", s.accuracy_exact)}${tile("sensitivity", s.sensitivity)}
        ${tile("specificity", s.specificity)}${tile("PPV", s.ppv)}${tile("NPV", s.npv)}
        ${a ? tile("alert sensitivity", a.sensitivity) + tile("false-alert rate", a.false_alert_rate) : ""}</div>
      ${a ? `<div class="meta">alerts: correct ${a.counts.alert_correct} · missed ${a.counts.alert_missed} · false ${a.counts.alert_false} · silent-correct ${a.counts.silent_correct}</div>` : ""}
      <div style="display:flex;gap:2rem;flex-wrap:wrap">${renderConfusion(s.confusion)}${renderPerClass(s.per_class)}</div>
    </div>`;
  });
  document.getElementById("perf-models").innerHTML = blocks.join("") ||
    `<div class="meta">No labelled events yet. Production data has no ground truth; curate a labelled set with cli.real_samples.</div>`;
  const tbody = document.querySelector("#perf-events tbody");
  tbody.innerHTML = data.events.map((e) => `<tr data-event-id="${esc(e.event_id)}">
    <td>${esc(fmtTime(e.processed_at))}</td><td>${esc(e.patient)}</td>
    <td><span class="oc oc-${esc(e.outcome)}">${esc(fmtOutcome(e.outcome))}</span></td>
    <td>${esc(e.predicted)} (${pct(e.confidence)})</td><td>${esc(e.truth)}</td>
    <td>${e.alert ? "✓" : "✗"} <span class="sub">${esc(e.reason_code)}</span></td>
    <td>${esc(e.source)}</td><td><span class="why" title="${esc(e.why)}">${esc(e.why)}</span></td></tr>`).join("");
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

// Theme toggle
document.getElementById("theme-btn").addEventListener("click", toggleTheme);
updateThemeButton();

// Tabs + performance view
document.getElementById("tabs").addEventListener("click", (e) => {
  const tab = e.target && e.target.dataset && e.target.dataset.tab;
  if (tab) showTab(tab);
});
document.getElementById("f-refresh").addEventListener("click", loadPerf);
["f-since", "f-dataset", "f-model"].forEach((id) =>
  document.getElementById(id).addEventListener("change", loadPerf));
document.querySelector("#perf-events tbody").addEventListener("click", (e) => {
  const tr = e.target.closest("tr");
  if (tr && tr.dataset.eventId) showEventInfo(tr.dataset.eventId, "perf-detail");
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
