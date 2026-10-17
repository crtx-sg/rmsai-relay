"""Render the rmsai-relay deployment diagram (Docker services, cloud vendors, call flow) to a PNG.

    uv run python docs/make_deploy_diagram.py   # writes deployment.png at the repo root

The logical pipeline view is docs/make_arch_diagram.py (architecture.png); this one shows what
runs where: the Compose services on the host, the cloud services, and the numbered path of one
event from ingest to the worklist and a phone call.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- palette -----------------------------------------------------------------------------------
C_HOST = "#f1f5f9"      # docker host band
C_CLOUD = "#fef2f2"     # cloud band (synthetic data only)
C_USER = "#ecfeff"      # clinician devices
C_APP = "#fae8ff"       # app containers (shared image)
C_STORE = "#ffffff"     # backing stores
C_REDIS = "#ffedd5"
C_LK = "#dbeafe"        # LiveKit
C_GUARD = "#fef9c3"     # de-id guard
C_VENDOR = "#ffffff"
DATA = "#1f2937"
BUS = "#ea580c"
VOICE = "#2563eb"
DBRW = "#16a34a"
CLOUD = "#b91c1c"
DASH = (0, (5, 3))

fig, ax = plt.subplots(figsize=(21, 13))
ax.set_xlim(0, 215)
ax.set_ylim(-6, 134)
ax.axis("off")


def band(x, y, w, h, color, label, sub=None, ec="none", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.5",
                                linewidth=1.6, edgecolor=ec, linestyle=ls, facecolor=color, zorder=0))
    ax.text(x + 1.5, y + h - 1.6, label, ha="left", va="top", fontsize=12, fontweight="bold",
            color="#334155", zorder=1)
    if sub:
        ax.text(x + 1.5, y + h - 4.6, sub, ha="left", va="top", fontsize=8.3, color="#475569",
                style="italic", zorder=1)


def box(x, y, w, h, title, body="", fc=C_STORE, ec="#334155", ls="-", fs=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                linewidth=1.4, edgecolor=ec, linestyle=ls, facecolor=fc, zorder=2))
    if body:
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=fs + 1.2,
                fontweight="bold", color="#111827", zorder=3)
        ax.text(x + w / 2, y + h - 5.2, body, ha="center", va="top", fontsize=fs, color="#1f2937",
                zorder=3, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs + 1.2,
                fontweight="bold", color="#111827", zorder=3)
    return dict(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2, r=x + w, t=y + h)


def arrow(p1, p2, color=DATA, style="-", lw=1.9, rad=0.0, both=False):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="<|-|>" if both else "-|>", mutation_scale=15,
                                 color=color, lw=lw, linestyle=style,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=4))


def path(points, color=DATA, style="-", lw=1.9, both=False):
    """Polyline with an arrowhead on the last segment (and the first, if `both`)."""
    xs, ys = zip(*points)
    ax.plot(xs[1:-1] if both else xs[:-1], ys[1:-1] if both else ys[:-1], color=color, lw=lw,
            linestyle=style, zorder=4, solid_capstyle="round")
    if both:
        arrow(points[1], points[0], color=color, style=style, lw=lw)
    arrow(points[-2], points[-1], color=color, style=style, lw=lw)


def label(x, y, text, color="#334155", fs=7.6, ha="center", bg=None):
    kw = dict(bbox=dict(boxstyle="round,pad=0.15", fc=bg, ec="none")) if bg else {}
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, style="italic", zorder=6, **kw)


def step(x, y, n, color=DATA):
    ax.text(x, y, n, ha="center", va="center", fontsize=10, fontweight="bold", color="white",
            zorder=7, bbox=dict(boxstyle="circle,pad=0.18", fc=color, ec="none"))


# --- title -------------------------------------------------------------------------------------
ax.text(107, 131, "rmsai-relay — deployment: Docker services, cloud services & call flow",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#0f172a")
ax.text(107, 127, "one Compose project on the host (network_mode: host)  ·  cloud services carry synthetic data only",
        ha="center", va="center", fontsize=10.5, color="#475569")

# --- bands -------------------------------------------------------------------------------------
band(2, 8, 23, 104, C_USER, "CLINICIAN")
band(28, 8, 119, 104, C_HOST, "DOCKER HOST", "PHI stays inside this box", ec="#64748b")
band(150, 8, 40, 104, C_CLOUD, "CLOUD", "synthetic / public data only", ec=CLOUD, ls=DASH)
band(193, 8, 20, 104, C_USER, "ON CALL")

# --- clinician devices -------------------------------------------------------------------------
browser = box(4, 78, 19, 20, "Browser", "companion app\nworklist · chat ·\npush-to-talk ·\nEvents tab", C_USER)
phone = box(195, 56, 16, 18, "Phone", "PSTN call\n(PIN → alert\n→ Q&A → ack)\n+ SMS", C_USER)

# --- app containers (one shared image, source bind-mounted) ------------------------------------
gw = box(31, 76, 21, 25, "gateway", ":8080  cli.gateway\n\nPIN /session · /ack\n/events · /metrics\n/artifact links\naudit log", C_APP)
vw = box(54, 76, 21, 25, "voice-worker", "agent rmsai-agent\n:8091 (local LiveKit)\n\ninbox room + WebRTC\ncalls: STT → PIN →\norchestrator → TTS", C_APP)
ing = box(77, 76, 21, 25, "tools  (one-shot)", "cli.ingest --emit bus\n\nHDF5 → SignalWindow\n→ ECGTransCovNet\n→ MEWS / trends\n→ DeviceEvent", C_APP, ls=DASH)
con = box(100, 76, 21, 25, "consumer", "cli.consume\nmain thread:\npersist · gate\nworklist push\nZADD call queue", C_APP, fs=7.4)
# the dispatcher is a thread inside the consumer process, not the voice worker
disp = box(101.4, 77.2, 18.2, 8.6, "CallDispatcher thread",
           "ZPOPMIN · stage alert\ndial · wait for hang-up", fc="#ede9fe", ec="#6d28d9", fs=6.2)
vwp = box(123, 76, 21, 25, "voice-worker-phone", "rmsai-agent-phone · :8082\nanswers; never dials\n\nper turn:\nSTT → PIN/wake gate\n→ orchestrator → TTS", C_APP, fs=7.2)

# --- backing stores ----------------------------------------------------------------------------
lk = box(31, 42, 21, 20, "livekit", ":7880 / 7881 / 7882udp\n\nlocal media server\ninbox room\nWebRTC call rooms", C_LK)
neo = box(54, 42, 21, 20, "neo4j", ":7687 / 7474\n\npatients · events\n(+ vital histories)\nreports · status\nbeds · protocols")
red = box(77, 42, 23, 20, "redis", ":6379 (in memory)\n\nstream rmsai.events\ncall queue (ZSET+HASH)\nstaged call alerts\nanswered-call keys (24 h)", C_REDIS)
qd = box(102, 42, 19, 20, "qdrant", ":6333\n\nclinical docs\nuploads\nreport narratives")
oll = box(123, 42, 21, 20, "Ollama", ":11434 on the host\n(or model-server,\nprofile llm)\n\nllama3.2:3b")

# --- in-process + volumes + optional -----------------------------------------------------------
box(31, 23, 56, 14, "app image: one for all app services (in-process, local)",
    "ECGTransCovNet (torch) · Presidio + spaCy de-id (DEID_BACKEND)\n"
    "BGE embeddings · Silero VAD · faster-whisper / Piper (local STT/TTS)", "#ffffff", fs=7.6)
box(90, 23, 54, 14, "./data  (bind mount)  +  named volumes",
    "HDF5 inputs · reports · ECG plots · kb_uploads · audit.jsonl\n"
    "neo4j_data · qdrant_data · hf_cache (model cache)", "#ffffff", fs=7.6)
box(31, 11, 113, 9, "optional profiles",
    "model-server Ollama (llm) · mosquitto MQTT (telemetry) · hapi-fhir EMR (emr) · real-samples (tools) · "
    "voice-worker-phone (telephony)", fc="#ffffff", ls=DASH, fs=7.2)

# --- cloud services ----------------------------------------------------------------------------
el = box(153, 92, 34, 13.5, "ElevenLabs", "STT Scribe · TTS flash\n(STT_BACKEND / TTS_BACKEND)", C_VENDOR, ec=CLOUD)
lkc = box(153, 77, 34, 14, "LiveKit Cloud", "SIP service · phone rooms\nrmsai-outbound-* / rmsai-call-*", C_LK, ec=CLOUD)
sw = box(153, 59, 34, 13, "SignalWire", "SIP trunk (TLS, digest)\nDomain App SWML → PSTN", C_VENDOR, ec=CLOUD)
tw = box(153, 45, 34, 11, "SignalWire SMS", "missed-call fallback\nsame number as the calls", C_VENDOR, ec=CLOUD)
cl = box(153, 26, 34, 14, "Cloud LLM (optional)", "Anthropic · Gemini · OpenAI-compat\nLLM_PROVIDER, prompts de-identified", C_VENDOR, ec=CLOUD, ls=DASH)

# =============================================================================================
# flows
# =============================================================================================
# clinician ↔ gateway (HTTP) and ↔ local LiveKit (WebRTC)
arrow((browser["r"], 92), (gw["x"], 92), both=True)
label(27.5, 94.5, "HTTPS", fs=7.2)
arrow((browser["cx"], browser["y"]), (lk["x"], lk["cy"] + 4), color=VOICE, both=True)
label(11.5, 66, "WebRTC\naudio + data", color=VOICE, fs=7.2)

# gateway → livekit: token + agent dispatch; gateway → neo4j: /events, /ack (⑦)
arrow((34, gw["y"]), (34, lk["t"]), color=DATA)
label(30.2, 67.5, "token +\ndispatch", fs=6.6, ha="right")
arrow((gw["r"] - 1, gw["y"]), (neo["x"] + 2, neo["t"]), color=DBRW, style=DASH)

# voice-worker ↔ local livekit (agent joins rooms); voice-worker reads the KB
arrow((vw["x"] + 2, vw["y"]), (lk["r"] - 1, lk["t"]), color=VOICE, both=True)
label(46.5, 69.5, "joins rooms", color=VOICE, fs=6.6, bg=C_HOST)
arrow((68.5, vw["y"]), (68.5, neo["t"]), color=DBRW, style=DASH)

# ① ingest → redis stream
arrow((ing["cx"], ing["y"]), (ing["cx"], red["t"]), color=BUS)
step(90.2, 64.3, "1", BUS)
label(84.5, 64.3, "XADD", color=BUS, fs=7, bg=C_HOST)

# ② consumer reads the stream, persists to neo4j + qdrant
arrow((con["x"] + 3, con["y"]), (red["r"] - 3, red["t"]), color=BUS, both=True)
step(98.6, 66.8, "2", BUS)
arrow((113, con["y"]), (113, qd["t"]), color=DBRW, style=DASH)
path([(106, con["y"]), (106, 71.5), (72.5, 71.5), (72.5, neo["t"])], color=DBRW, style=DASH)
label(116.6, 64.3, "persist", color=DBRW, fs=7, bg=C_HOST)

# ③ consumer → local livekit: worklist push (server API send_data), at once
path([(101.8, con["y"]), (101.8, 74), (41, 74), (41, lk["t"])], color=DATA)
step(78.5, 74, "3")
label(61, 72.6, "worklist push (server API)", fs=6.4, bg=C_HOST)

# ④ dispatcher: queue in redis, stage alert, then dispatch + dial on LiveKit Cloud
path([(disp["r"], disp["cy"]), (122, disp["cy"]), (122, 108), (148, 108), (148, 86),
      (lkc["x"], 86)], color=VOICE)
step(119, 108, "4", VOICE)
label(121.5, 110.2, "dispatch agent + SIP dial",
      color=VOICE, fs=7, ha="left")

# ⑤ LiveKit Cloud ↔ SignalWire ↔ phone
arrow((lkc["cx"], lkc["y"]), (sw["cx"], sw["t"]), color=VOICE, both=True)
label(lkc["cx"] + 5.5, 74.5, "SIP / TLS", color=VOICE, fs=7)
arrow((sw["r"], sw["cy"]), (phone["x"], phone["cy"]), color=VOICE, both=True)
step(190.5, 69.5, "5", VOICE)
label(190.5, 61.5, "PSTN", color=VOICE, fs=7)

# ⑥ phone worker ↔ LiveKit Cloud (audio), + redis alert, KB, LLM
arrow((vwp["r"], 81), (lkc["x"], 81), color=VOICE, both=True)
step(148.5, 78, "6", VOICE)
arrow((vwp["cx"], vwp["y"]), (vwp["cx"], oll["t"]), color=DATA)
label(vwp["cx"] + 1.2, 66.6, "LLM (de-id first)", fs=6.6, ha="left", bg=C_HOST)
path([(126, vwp["y"]), (126, 69), (92, 69), (92, red["t"])], color=BUS, style=DASH)
label(121, 69, "staged alert", color=BUS, fs=6.6, bg=C_HOST)

# both workers → ElevenLabs through the Presidio guard (TTS text); STT sends raw audio
guard = box(100, 114, 34, 6, "Presidio de-id → TTS (text to speak)", fc=C_GUARD, ec="#a16207", fs=6.8)
path([(64.5, vw["t"]), (64.5, 117), (guard["x"], 117)], color=CLOUD)
arrow((131, vwp["t"]), (131, guard["y"]), color=CLOUD)
path([(guard["r"], 117), (178, 117), (178, el["t"])], color=CLOUD)
label(67, 118.8, "TTS text from both workers", color=CLOUD, fs=6.8, ha="left")
# STT: the phone worker sends the caller's raw audio (no de-id possible) and gets text back
arrow((vwp["r"], 97.5), (el["x"], 97.5), color=CLOUD, style=DASH, both=True)
label(148.6, 100.4, "STT", color=CLOUD, fs=6.6, bg=C_CLOUD)
label(170, 94.6, "STT gets raw audio: synthetic speech only", color=CLOUD, fs=6.6)

# SMS fallback; optional cloud LLM instead of Ollama
path([(119, con["y"]), (119, 72.2), (149.2, 72.2), (149.2, 50.5), (tw["x"], 50.5)], color=DATA, style=DASH)
ax.text(150.7, 62, "SMS fallback", rotation=90, ha="center", va="center", fontsize=6.4, color="#334155", style="italic", zorder=6)
arrow((tw["r"], tw["cy"]), (phone["x"] + 2, phone["y"]), color=DATA, style=DASH)
arrow((oll["r"], oll["cy"] - 4), (cl["x"], cl["cy"] + 2), color="#64748b", style=DASH)
label(148.5, 43.8, "or", color="#64748b", fs=7)

# ⑦ acknowledge closes the loop (app Ack via the gateway, or "acknowledge" on the call)
step(54.6, 66.3, "7", DBRW)
step(197, 76, "7", DBRW)

# --- numbered flow caption ---------------------------------------------------------------------
caption = (
    "①  cli.ingest classifies the HDF5 event and XADDs it to the Redis stream.   "
    "②  consumer reads it, persists to Neo4j + Qdrant and runs the alert gate.   "
    "③  every gated event is pushed to the app worklist at once (LiveKit server API → inbox room).\n"
    "④  the call is queued (Redis ZSET, most severe first); the consumer's CallDispatcher thread stages the alert and dials via LiveKit Cloud, one call at a time.\n"
    "⑤  SignalWire rings the phone.   "
    "⑥  the phone worker answers: per turn, caller audio → STT (ElevenLabs) → PIN/wake gate → orchestrator (KB, Ollama) → Presidio → TTS (ElevenLabs) → audio back.\n"
    "⑦  \"acknowledge\" (phone) or Ack (app) sets the event's status in Neo4j; a queued call for an acknowledged event is skipped.   "
    "In the app, the voice-worker serves the inbox room the same way over the local LiveKit."
)
ax.text(107.5, -2.2, caption, ha="center", va="center", fontsize=8.1, color="#334155", linespacing=1.6)

# --- legend ------------------------------------------------------------------------------------
handles = [plt.Line2D([0], [0], color=DATA, lw=2.4, label="HTTP / control"),
           plt.Line2D([0], [0], color=BUS, lw=2.4, label="Redis"),
           plt.Line2D([0], [0], color=VOICE, lw=2.4, label="audio / SIP"),
           plt.Line2D([0], [0], color=DBRW, lw=2.4, ls=DASH, label="KB read / write"),
           plt.Line2D([0], [0], color=CLOUD, lw=2.4, label="leaves the host")]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(2.6, 10), bbox_transform=ax.transData,
          fontsize=7.0, frameon=True, ncol=1, title="arrows", title_fontsize=7.5, handlelength=1.6)

out = Path(__file__).resolve().parents[1] / "deployment.png"
fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")  # vector, for slides
print(f"wrote {out} + .svg")
