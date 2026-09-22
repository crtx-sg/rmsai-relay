"""Render the rmsai-relay architecture / data-flow diagram to a PNG (matplotlib, no graphviz).

    uv run python docs/make_arch_diagram.py   # writes architecture.png at the repo root
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

# --- palette -----------------------------------------------------------------------------------
C_EVENT = "#dbeafe"     # event lane bg (light blue)
C_CONV = "#dcfce7"      # conversation lane bg (light green)
C_DB = "#f1f5f9"        # shared knowledge bg (gray)
C_BOX = "#ffffff"
C_MODEL = "#fef9c3"     # ML/model boxes (yellow)
C_BUS = "#ffedd5"       # redis (orange)
C_DISP = "#fae8ff"      # dispatch (violet)
DATA = "#1f2937"        # data flow (near-black)
BUS = "#ea580c"         # event bus (orange)
VOICE = "#2563eb"       # voice/audio (blue)
DBRW = "#16a34a"        # db read/write (green)
DASH = (0, (5, 3))

fig, ax = plt.subplots(figsize=(19, 11.5))
ax.set_xlim(0, 192)
ax.set_ylim(0, 116)
ax.axis("off")


def band(x, y, w, h, color, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.5",
                                linewidth=0, facecolor=color, zorder=0))
    ax.text(x + 1.8, y + h - 2.0, label, ha="left", va="top", fontsize=12.5,
            fontweight="bold", color="#334155", zorder=1)


def box(x, y, w, h, text, fc=C_BOX, fs=9.5, ec="#334155"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", zorder=3, color="#111827")
    return dict(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2, top=(x + w / 2, y + h),
                bot=(x + w / 2, y), left=(x, y + h / 2), right=(x + w, y + h / 2))


def note(x, y, text, fs=8, color="#475569"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color, style="italic", zorder=3)


def arrow(p1, p2, color=DATA, style="-", lw=2.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=18, color=color, lw=lw,
                                 linestyle=style, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=3, shrinkB=3, zorder=4))


def biarrow(p1, p2, color=VOICE, lw=2.2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="<|-|>", mutation_scale=15, color=color, lw=lw,
                                 connectionstyle="arc3,rad=0", shrinkA=3, shrinkB=3, zorder=4))


# --- title -------------------------------------------------------------------------------------
ax.text(96, 112.5, "rmsai-relay — architecture & pipeline flow", ha="center", va="center",
        fontsize=19, fontweight="bold", color="#0f172a")
ax.text(96, 107.5, "physiological event  →  arrhythmia classification  →  knowledge base  →  clinician alert & grounded Q&A",
        ha="center", va="center", fontsize=11, color="#475569")

# --- lanes -------------------------------------------------------------------------------------
band(2, 72, 188, 31, C_EVENT, "EVENT PIPELINE   (device → alert)")
band(2, 39, 188, 29, C_DB, "SHARED KNOWLEDGE BASE   &   DISPATCH")
band(2, 2, 188, 33, C_CONV, "CONVERSATION PIPELINE   (clinician Q&A, voice or text)")

# --- event pipeline (top, left→right) ----------------------------------------------------------
e1 = box(5, 81, 22, 14, "Device\nHDF5 file / MQTT")
e2 = box(31, 81, 22, 14, "Ingest\n→ SignalWindow\n(ECG+vitals+history)", fs=8.8)
e3 = box(57, 81, 24, 14, "ECG Model\nECGTransCovNet\n(CNN+Transformer, ML)", C_MODEL, fs=8.8)
e4 = box(85, 81, 24, 14, "Vitals analysis\nMEWS (rules) +\nMann-Kendall (stats)", fs=8.8)
e5 = box(113, 81, 24, 14, "FP gate + criticality\n+ care guidance\n→ DeviceEvent", fs=8.8)
e6 = box(141, 82.5, 16, 11, "Redis Stream\n(event bus)", C_BUS, fs=8.8)
e7 = box(161, 81, 26, 14, "Consumer\npersist + decide\n(should_call?)", fs=8.8)

note(69, 79.3, "→ 1 arrhythmia class + confidence")
note(42, 79.3, "no diagnosis yet")

arrow(e1["right"], e2["left"]); arrow(e2["right"], e3["left"]); arrow(e3["right"], e4["left"])
arrow(e4["right"], e5["left"])
arrow(e5["right"], e6["left"], color=BUS); arrow(e6["right"], e7["left"], color=BUS)

# --- hand-off caption (in the gap between event lane and middle lane) --------------------------
note(96, 70, "Event pipeline  WRITES  the knowledge base   ·   Conversation pipeline  READS  it   (the hand-off)",
     fs=9.5, color="#334155")

# --- shared databases (middle, right) ----------------------------------------------------------
db1 = box(114, 44, 32, 13, "Neo4j  (graph)\npatient · event · report ·\nconditions · beds", fs=8.4)
db2 = box(150, 44, 34, 13, "Qdrant  (vector)\nreport-narrative\nembeddings", fs=8.4)

# consumer writes to both DBs
arrow((e7["cx"] - 4, e7["y"]), db1["top"], color=DBRW, style=DASH, rad=0.2)
arrow((e7["cx"] + 4, e7["y"]), db2["top"], color=DBRW, style=DASH, rad=0.0)
note(168, 62.5, "persist (write)", color=DBRW)

# --- dispatch (middle, left): should_call → two delivery channels ------------------------------
d1 = box(6, 44, 46, 13, "App worklist push  →  inbox room\ndata message (LiveKit server API)\nshown in companion app", C_DISP, fs=8.2)
d2 = box(56, 44, 48, 13, "Outbound call  →  per-event room\nSIP dial (phone) / WebRTC link\nalert staged in Redis", C_DISP, fs=8.2)

arrow((e7["x"] + 2, e7["y"] + 3), (d2["x"] + d2["w"], d2["cy"] + 3), color=DATA, rad=0.12)
note(120, 66, "should_call?  (criticality ≥ High,\narrhythmia confidence, vitals override)", color="#334155")

# --- conversation pipeline (bottom, left→right) ------------------------------------------------
c1 = box(5, 10, 22, 17, "Clinician\nphone (SIP) /\nbrowser (WebRTC) /\ncompanion app", fs=8.6)
c2 = box(34, 10, 20, 17, "LiveKit Server\ntransport only\n(WebRTC / SIP)", fs=8.6)
c3 = box(61, 8, 38, 21, "Voice Worker  (agent)\njoins ONE room\n\nSTT → VAD → gates (PIN/wake)\n→ Handler → TTS", fs=8.8)
c4 = box(106, 10, 42, 17, "Orchestrator  (RAG turn)\nguardrails → retrieve (vector+graph)\n→ de-id → LLM → guardrails", fs=8.4)

note(80, 5.2, "STT: Whisper / ElevenLabs   ·   TTS: Piper / ElevenLabs   ·   VAD: Silero", fs=7.8)
note(127, 6.6, "LLM: Ollama llama3.2   ·   embeddings: BGE / hashing   ·   de-id: Presidio / regex", fs=7.8)

# clinician <-> LiveKit: audio both ways (WebRTC or SIP)
biarrow(c1["right"], c2["left"])
note(30.5, 21, "audio", fs=7.2, color=VOICE)

# LiveKit <-> worker: two explicit one-way audio tracks, BOTH in the same joined room
arrow((c3["x"], c3["cy"] + 3.5), (c2["x"] + c2["w"], c2["cy"] + 3.5), color=VOICE)  # publish TTS
arrow((c2["x"] + c2["w"], c2["cy"] - 3.5), (c3["x"], c3["cy"] - 3.5), color=VOICE)  # subscribe speech
note(57.5, 25.6, "◀ TTS", fs=7.2, color=VOICE)
note(57.5, 11.4, "speech ▶", fs=7.2, color=VOICE)

# worker <-> orchestrator (in-process)
arrow(c3["right"], (c4["x"], c4["cy"] + 2), color=DATA)
arrow((c4["x"], c4["cy"] - 3), (c3["x"] + c3["w"], c3["cy"] - 3), color=DATA, rad=0.0)
note(102.5, 12.6, "answer", fs=7.2, color="#334155")

# orchestrator reads the DBs
arrow(c4["top"], (db1["cx"] - 6, db1["y"]), color=DBRW, style=DASH, rad=-0.12)
note(133, 39, "retrieve (read)", color=DBRW)

# callout: the worker is a two-way participant within ONE room
_cx, _cy, _cw, _ch = 151, 8, 37, 19
ax.add_patch(FancyBboxPatch((_cx, _cy), _cw, _ch, boxstyle="round,pad=0.3,rounding_size=1.2",
                            linewidth=1.3, edgecolor=VOICE, facecolor="#eff6ff", zorder=2))
ax.text(_cx + _cw / 2, _cy + _ch - 2.3, "Two-way audio in ONE room", ha="center", va="top",
        fontsize=8.8, fontweight="bold", color=VOICE, zorder=3)
ax.text(_cx + 2.2, _cy + _ch - 6.0,
        "the worker joins one room —\ninbox (app) OR per-event (call) —\nand within that room it:\n"
        "▼ PUBLISHES its TTS  → caller hears\n▲ SUBSCRIBES to caller audio → STT",
        ha="left", va="top", fontsize=7.5, color="#1e3a5f", zorder=3, linespacing=1.55)

# dispatch → conversation
arrow(d1["bot"], (c1["cx"], c1["y"] + c1["h"]), color=DATA, rad=-0.12)
arrow(d2["bot"], (c2["cx"], c2["y"] + c2["h"]), color=VOICE, rad=0.0)
note(20, 37, "worklist", fs=8, color=DATA)
note(60, 37, "place call", fs=8, color=VOICE)

# --- legends -----------------------------------------------------------------------------------
box_legend = [Patch(facecolor=C_MODEL, edgecolor="#334155", label="ML / model"),
              Patch(facecolor=C_BUS, edgecolor="#334155", label="Redis"),
              Patch(facecolor=C_DISP, edgecolor="#334155", label="dispatch")]
leg1 = ax.legend(handles=box_legend, loc="upper left", bbox_to_anchor=(0.006, 0.985),
                 fontsize=9, frameon=True, title="boxes", title_fontsize=9)
ax.add_artist(leg1)

arr_legend = [plt.Line2D([0], [0], color=DATA, lw=2.4, label="data flow"),
              plt.Line2D([0], [0], color=BUS, lw=2.4, label="event bus (Redis)"),
              plt.Line2D([0], [0], color=VOICE, lw=2.4, label="voice / audio call"),
              plt.Line2D([0], [0], color=DBRW, lw=2.4, ls=DASH, label="KB read / write")]
ax.legend(handles=arr_legend, loc="upper right", bbox_to_anchor=(0.995, 0.985),
          fontsize=9, frameon=True, title="arrows", title_fontsize=9)

out = Path(__file__).resolve().parents[1] / "architecture.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
