# Configuration

Every setting and stub, and how the call decision, per-hospital files and vital trends are configured. Install first: [INSTALL.md](INSTALL.md).

## POC configuration & stubs

Everything synthetic/simplified sits behind an interface so the real implementation swaps in without
touching callers.

**Key config defaults** (all env/`.env`-overridable; the app reads the repo-root `.env` itself.
Keys marked *Compose* are Docker Compose settings, which Compose does **not** read from that file;
see [Two `.env` readers](INSTALL.md#two-env-readers)):

Keys marked *(hospital)* are per-hospital policy: set them in `config/hospitals/<HOSPITAL_ID>.yaml`
(see [Per-hospital configuration](#per-hospital-configuration)). The environment variable still works,
but only as a logged override.

| Key | Default | Purpose |
|-----|---------|---------|
| `FP_SUPPRESS_MIN_CONFIDENCE` *(hospital)* | `0.80` | suppress as FP only if `NORMAL_SINUS` ≥ this; else flag `uncertain` |
| `LOW_CONFIDENCE_CAVEAT` *(hospital)* | `0.60` | top-class confidence below this marks the event `low_confidence` |
| `CRITICALITY_NORMAL_EVENT` *(hospital)* | `NORMAL_SINUS` | the only event treated as non-critical; any other event ⇒ at least High |
| `CRITICALITY_MEWS_THRESHOLD` *(hospital)* | `3` | MEWS score at/above this ⇒ escalate criticality to High |
| `CRITICALITY_ESCALATE_ON_DETERIORATING` *(hospital)* | `true` | any deteriorating vital trend ⇒ escalate criticality to High |
| `CRITICALITY_FP_OVERRIDE_ON_VITALS` *(hospital)* | `true` | call even on a confident false-positive ECG (NORMAL_SINUS) when vitals warrant it (MEWS ≥ threshold or deteriorating); overrides the spec-D10 no-call guard |
| `DISPATCH_MODE` *(hospital)* | `app+call` | where a gated event goes: `app` (worklist push only), `call` (voice/text only), `app+call` |
| `HOSPITAL_ID` | *(empty)* | scopes the app inbox room `rmsai-inbox-<id>` and selects `config/hospitals/<id>.yaml` |
| `OUTBOUND_ENABLED` / `OUTBOUND_MIN_CRITICALITY` *(hospital)* | `false` / `High` | read by `should_call`, but **`cli.consume` and `cli.outbound` force `OUTBOUND_ENABLED` on** and take the threshold from `--min-criticality` (default `High`). The gate governs both the worklist push and the call |
| `OUTBOUND_MIN_ARRHYTHMIA_CONFIDENCE` *(hospital)* | `0.60` | a non-normal (arrhythmia) event is only asserted *as a rhythm* if confidence is at/above this. Below it: withheld when the vitals are calm, or re-based as a **vitals-driven alert** (rhythm marked unconfirmed) when they aren't. Vitals never raise the bar |
| `OUTBOUND_CALL_VITALS_ALERTS` *(hospital)* | `false` | phone a vitals-driven alert (rhythm below the confidence gate, vitals warrant it). `false` = worklist only; the consumer logs `no call: vitals-driven alert … reported to the app only` |
| `OUTBOUND_CALL_NUMBER` / `OUTBOUND_FROM` *(hospital)* | — | destination (the default for `--number` on `cli.consume`/`cli.outbound`, and for `cli.call`) + caller ID (the SignalWire number; also the SMS sender). A real call/SMS with neither `--number` nor `OUTBOUND_CALL_NUMBER` refuses to start; a placeholder is used only for simulated runs |
| `OUTBOUND_MAX_RETRIES` / `OUTBOUND_RETRY_DELAY_S` *(hospital)* | `2` / `30` | no-answer retry policy: 1 + `max_retries` attempts, each redial in its own room, then the SMS fallback. A pickup with no PIN counts as no answer; a PIN lockout ends it with no retry and no SMS |
| `INBOUND_AUTH_PIN` | shared PIN | verified before any PHI is voiced; spoken or keyed on a phone (4 digits or `#`, `*` starts over). 3 wrong tries lock that call out (the count resets every call) |
| `AUDIO_WAKE_WORD` | `hey vios` | wake word that gates follow-up *audio* Q&A on a call (text chat is never gated) |
| `AUDIO_WAKE_WINDOW_S` | `30` | seconds the agent stays "awake" after a wake word so audio follow-ups needn't repeat it (audio only — does not affect text chat) |
| `AUDIO_WAKE_REQUIRED` | `true` | require the wake word to open a follow-up audio turn (SIP/playground Q&A). Set `false` to answer **every** authenticated audio turn — the escape hatch when STT mishears the out-of-vocab brand word. The companion app already bypasses the gate (it controls the mic) |
| `INBOX_SPEAK_ON_SELECT` | `true` | companion app: selecting a worklist row speaks that event's stored report summary aloud (in addition to scoping chat). Spoken text is the `Report.summary` — no model call. `false` turns it off everywhere. When on, the app's header toggle (🔊 Voice on select / 🔇 Voice muted) controls it per browser. Its default is **muted when `dispatch_mode` is `app+call`**, so a clinician on the phone alert doesn't hear the same event twice; it's remembered per dispatch mode. A muted selection is sent as `/select-silent`, so no TTS runs at all. Push-to-talk answers are always spoken |
| `LIVEKIT_REDISPATCH_ON_START` | `true` | on worker startup, auto re-dispatch the agent into live `rmsai-inbox-*` rooms that lost their agent (e.g. after a worker restart), so the app doesn't need a re-login. On-demand equivalent: `cli.dispatch` |
| `LIVEKIT_WORKER_HTTP_PORT` | `8081` | port for livekit-agents' health-check HTTP server. Only matters when two workers run side by side. The Docker `voice-worker` gets `8091` from Compose (*Compose*: set in the shell to change it), because host networking would otherwise collide with a worker run by hand (`[errno 98] address already in use`). The phone worker uses `LIVEKIT_SIP_WORKER_HTTP_PORT` (`8082`) |
| `GATEWAY_PORT` | `8080` | *Compose*: host port for the containerized gateway. Move it (`GATEWAY_PORT=8090 make docker-up`) if `hapi-fhir` (`--profile emr`) or a hand-run gateway already holds 8080 |
| `LLM_PROVIDER` / `LLM_MODEL` / `OLLAMA_URL` | `echo` / `llama3.2` / `http://localhost:11434` | `echo` is offline and deterministic; `ollama` for self-hosted answers (`LLM_MODEL` is the Ollama model); `anthropic` / `gemini` / `openai` for cloud models, see [Choosing the LLM](#choosing-the-llm). An unknown name is an error |
| `EMBEDDER` / `KB_MIN_RELEVANCE` / `KB_LLM_ROUTER` / `KB_UPLOAD_DIR` | `hashing` / `0.60` / `false` / `data/kb_uploads` | KB embedder (must match the collection), relevance floor for grounded answers, LLM question router, managed upload folder |
| `STT_BACKEND` / `TTS_BACKEND` | `stub` / `stub` | `whisper`/`piper` (self-hosted) or `elevenlabs` (cloud, synthetic data only) |
| `LIVEKIT_AGENT_NAME` / `LIVEKIT_CALL_ROOM_PREFIX` | `rmsai-agent` / `rmsai-call-` | the app worker's dispatch name; prefix of on-demand and inbound phone-call rooms |
| `SIP_RINGING_TIMEOUT_S` / `SIP_MAX_CALL_DURATION_S` | `30` / `600` | call safety rails (unanswered ring time; hard cap on a call) |
| `LIVEKIT_SIP_*`, `LIVEKIT_SIP_TRUNK_ID` | *(empty)* | phone calls (who may call in is per hospital: `outbound.inbound_allowed_numbers`): see [Phone calls](TELEPHONY_SETUP.md#phone-calls-livekit-cloud--signalwire) |
| `SIGNALWIRE_SPACE` / `_PROJECT_ID` / `_API_TOKEN` | *(empty)* | real SMS via SignalWire (`--notifier signalwire`), sent from `outbound.from`: the text channel and the unanswered-call fallback |
| `EPISODIC_RECALL` | `false` | condition free-text answers on recalled cross-session past Q&A; off keeps answers grounded in the live KB + current conversation only |
| `STT_LANGUAGE` | `en` | force the STT language (ISO 639-1); blank/`auto` = auto-detect. Stops Whisper/Scribe "hearing" other languages on noise |
| `ECG_CHECKPOINTS` | *(unset)* | ECG classifier checkpoint path(s), comma- or space-separated. Unset ⇒ the deterministic stub. Several paths load as one softmax-averaging ensemble; `external/ecgtranscnn/models/real_v2/fold{0..4}.pt` is the recommended artifact (run `make weights` first). Head, lead order and filter preset are read from the checkpoint |
| `INGEST_TIME_ANCHOR` | `source` | `source` keeps the HDF5 event times; `now` shifts each patient's events so the latest is the ingest time (`cli.ingest --time-anchor`) |
| `ECG_PLOT_ENABLED` / `PLOT_DIR` | `true` / `data/plots` | producer renders each event's ECG lead to `{PLOT_DIR}/<event_id>.png` (gitignored); path persisted as `MonitoredEvent.ecg_plot_ref` |
| `DEID_BACKEND` | `regex` | `regex` (offline) / `presidio` / `auto` (presidio if installed, else regex) |

**Stubs (POC → production):** `PatientHistory` (synthetic, seeded by `patient_id` → EMR/FHIR) ·
`BedAssignment` (≤25 beds/unit, overflow, clear ops → ADT feed) · `ECGModel` (vendored `real_v2`
research checkpoint, or the deterministic test stub when none is configured → validated SaMD model
on Triton) · `EventStore` (Neo4j → Postgres/
TimescaleDB at scale) · FHIR client (stub → HAPI) · outbound (single number → escalation tree) ·
caller auth (shared PIN → per-user identity/MFA) · audit (JSONL → tamper-evident store).

### Choosing the LLM

The LLM writes the grounded answer for KB questions (chat and voice) and, with `KB_LLM_ROUTER=true`,
picks the graph template. Retrieval itself (BGE embeddings, Qdrant, Neo4j) does not use it. Pick one
with `LLM_PROVIDER`:

| `LLM_PROVIDER` | Model setting (default) | Credentials | Notes |
|---|---|---|---|
| `echo` | — | — | deterministic, offline; tests |
| `ollama` | `LLM_MODEL` (`llama3.2`) | — | self-hosted (`OLLAMA_URL`); the only option for real PHI |
| `anthropic` | `ANTHROPIC_MODEL` (`claude-opus-5-5`) | `ANTHROPIC_API_KEY` | `ANTHROPIC_EFFORT` (`low`) trades thinking depth for latency |
| `gemini` | `GEMINI_MODEL` (`gemini-2.5-flash`) | `GEMINI_API_KEY` | via Google's OpenAI-compatible endpoint (`GEMINI_BASE_URL`) |
| `openai` | `OPENAI_MODEL` (`gpt-4o-mini`) | `OPENAI_API_KEY` | **any OpenAI-compatible API**: set `OPENAI_BASE_URL` for Mistral, Groq, OpenRouter, Together, or a local vLLM / LM Studio server |

```bash
# .env, e.g. Claude:
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
# or Groq through the OpenAI-compatible provider:
LLM_PROVIDER=openai
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=gsk_...
OPENAI_MODEL=<model id from the provider>
```

Then restart what uses it (`make docker-restart`, or the host CLIs). Try it without the app:
`uv run python -m cli.kb "what is the SOP for atrial fibrillation?"` or
`uv run python -m cli.text_chat --llm anthropic`.

- **Install.** Cloud providers use the official `anthropic` / `openai` SDKs from the `llm-cloud` extra,
  included in `make setup-all` and the Docker image. After pulling this change, rebuild the image once
  (`make docker-build && make docker-up`).
- **Safety.** Cloud providers are for **synthetic data only** (hard rules 4 and 5). Every provider
  sits behind `DeidentifyingLLM`, so the prompt is de-identified (`DEID_BACKEND`) before it leaves,
  and selecting a cloud provider prints a warning at startup. Real patients stay on `ollama`.
- **Answers are short and often spoken**, so each request carries a one-line latency instruction.
  `LLM_MAX_TOKENS` caps the answer; the default `0` leaves it to the provider (OpenAI reasoning models
  reject the parameter and smaller hosted models cap lower), except Anthropic, which requires a cap
  and gets 16000. `LLM_TIMEOUT_S` (60) bounds a request. If a call fails (no credit,
  rate limit, network), the orchestrator retries and then falls back to a safe answer instead of
  crashing the turn.
- **Claude specifics.** Thinking is adaptive; `ANTHROPIC_EFFORT=low` keeps time-to-first-word down for
  voice (raise to `medium`/`high` for harder questions; `none` sends no effort, and Haiku or older
  models, which reject it, get none automatically). On current models a refused request is
  re-run server-side on Anthropic's recommended fallback model; a refusal that still stands returns
  "The model declined to answer that." When streaming (voice), words already spoken can't be
  recalled, so that sentence is appended to mark where the answer stopped.
- **Model names change.** The defaults above are starting points; check the provider's model list and
  set the `*_MODEL` variable.

### Criticality & the outbound-call decision

Criticality (`common/criticality.py`) drives both the persisted `MonitoredEvent.criticality` and the
outbound-call gate, and all of its inputs are configurable (table above). It is computed in layers:

1. **Intrinsic** — `criticality(event_type, mews_risk)` takes the more severe of the arrhythmia
   class and the MEWS risk (so VT/VF/ST_ELEVATION are intrinsically `Critical`).
2. **Configurable escalation** — `event_criticality(event, config)` raises that to **at least
   `High`** when **any** of:
   - the event is **not** the normal baseline (`CRITICALITY_NORMAL_EVENT`, default `NORMAL_SINUS`) —
     i.e. *any* real arrhythmia is at least High;
   - the **MEWS score ≥ `CRITICALITY_MEWS_THRESHOLD`** (default 3);
   - a **vital is deteriorating** (clinically significant trend, see
     [Vital trends](#vital-trends-when-is-a-vital-deteriorating)), when
     `CRITICALITY_ESCALATE_ON_DETERIORATING` is on.

   Escalation only ever raises to `High` — it never lowers an already-`Critical` event.
3. **Call gate** (`should_call`) — dials out when `outbound_enabled` and the criticality is at or
   above `OUTBOUND_MIN_CRITICALITY` (default `High`). The event is **always persisted**; the gate
   governs both the worklist push and the call.

**Arrhythmia confidence gate (the model must be sure — or say it isn't).** A **non-normal
(arrhythmia)** prediction is only asserted *as a rhythm* when the model's confidence in it is at/above
`OUTBOUND_MIN_ARRHYTHMIA_CONFIDENCE` (default `0.60`). Below that, what happens next depends on the
patient, not on the classifier:

- **vitals unremarkable** → withheld (`low_confidence_arrhythmia (45% < 60%)`). The event is still
  persisted and answerable in chat; only the alert is suppressed.
- **vitals warrant attention** (MEWS ≥ threshold or a deteriorating trend) → the alert still goes out
  to the app worklist (and is phoned only when the hospital sets `outbound.call_vitals_alerts: true`;
  default `false`), but **re-based on the vitals** (`vitals_alert (MEWS 4 >= threshold 3)`). The worklist row leads with
  *Vitals alert* and the named vital, the rhythm is demoted to `unconfirmed: <type>`, and the spoken
  call alert says "Possible …, treat the rhythm as unconfirmed".

Vitals never raise the confidence bar — they cannot make an uncertain classification true — they only
change what the alert **claims**. `common.criticality.alert_basis` decides that once, and every
surface leads with it. Rows also carry their confidence, flagged below `LOW_CONFIDENCE_CAVEAT`.
See the flowchart in [ARCHITECTURE.md](ARCHITECTURE.md) or [`docs/alert-gate.html`](docs/alert-gate.html).

**False-positive override (vitals beat the rhythm).** A confident `NORMAL_SINUS`
(≥ `FP_SUPPRESS_MIN_CONFIDENCE`) is a false positive and normally does **not** call (spec D10). When
`CRITICALITY_FP_OVERRIDE_ON_VITALS` is on (default), a **vitals-driven escalation** (MEWS ≥ threshold
or deteriorating) **overrides** that guard — the patient is deteriorating regardless of the rhythm,
so the call still fires. This case is made explicit at three layers so it is never confusing:

- **decision reason** → `fp_override (MEWS 5 >= threshold 3)` / `fp_override (vitals deteriorating)`
  (also written to the audit log);
- **console** → `[consume] FALSE-POSITIVE OVERRIDE: ECG classified NORMAL_SINUS (false positive),
  but the patient's vitals warrant a call … — vitals/MEWS-driven escalation, not the rhythm.`;
- **spoken alert** → appends *"Note: the ECG rhythm is classified as normal sinus, so this alert is
  driven by the patient's vitals (…), not the rhythm."*

Set `CRITICALITY_FP_OVERRIDE_ON_VITALS=false` to revert to the strict spec-D10 behaviour
(NORMAL_SINUS ⇒ never call).

### Per-hospital configuration

Alerting policy differs by site, so it lives in one YAML file per hospital rather than in `.env`:

```
config/hospitals/
  default.yaml        every setting, the defaults (committed)
  h1.example.yaml     template for a site file (committed, placeholder numbers)
  h1.yaml             the site's own values (gitignored: it holds real phone numbers)
```

`HOSPITAL_ID` in `.env` picks the file (`h1` → `h1.yaml`); `HOSPITAL_CONFIG_DIR` moves the directory.
Settings combine in this order, each overriding the one before:

1. `default.yaml`
2. `<HOSPITAL_ID>.yaml`, key by key (list only what differs, or everything to keep it in one place)
3. an environment variable (shell or `.env`), meant for a quick test. Every such override is logged at
   startup by name, never by value: `[config] environment overrides hospital config (default.yaml +
   h1.yaml): OUTBOUND_CALL_NUMBER`

| Section | YAML key | Environment override |
|---|---|---|
| `escalation` | `normal_event`, `mews_threshold`, `escalate_on_deteriorating`, `fp_override_on_vitals` | `CRITICALITY_NORMAL_EVENT`, `CRITICALITY_MEWS_THRESHOLD`, `CRITICALITY_ESCALATE_ON_DETERIORATING`, `CRITICALITY_FP_OVERRIDE_ON_VITALS` |
| `escalation` | `fp_suppress_min_confidence`, `low_confidence_caveat` | `FP_SUPPRESS_MIN_CONFIDENCE`, `LOW_CONFIDENCE_CAVEAT` |
| `outbound` | `dispatch_mode`, `enabled`, `min_criticality`, `min_arrhythmia_confidence`, `call_vitals_alerts` | `DISPATCH_MODE`, `OUTBOUND_ENABLED`, `OUTBOUND_MIN_CRITICALITY`, `OUTBOUND_MIN_ARRHYTHMIA_CONFIDENCE`, `OUTBOUND_CALL_VITALS_ALERTS` |
| `outbound` | `call_number`, `from`, `inbound_allowed_numbers` | `OUTBOUND_CALL_NUMBER`, `OUTBOUND_FROM`, `SIP_INBOUND_ALLOWED_NUMBERS` |
| `outbound` | `max_retries`, `retry_delay_s` | `OUTBOUND_MAX_RETRIES`, `OUTBOUND_RETRY_DELAY_S` |
| `vitals_trends` | `alpha`, `vitals.<name>.{min_change, normal, unit}` | — (see [Vital trends](#vital-trends-when-is-a-vital-deteriorating)) |

**A new site:** `cp config/hospitals/h1.example.yaml config/hospitals/<id>.yaml`, edit it, set
`HOSPITAL_ID=<id>` in `.env`, then `make docker-restart` (the repo is bind-mounted, so the containers
see the file without a rebuild).

**Validation** stops startup with a message naming the key: phone numbers must be **quoted** E.164
(`"+15551234567"`; unquoted, YAML reads `+91…` as an integer and drops the `+`), `dispatch_mode` must be
`app`, `call` or `app+call`, `min_criticality` one of `Low`/`Medium`/`High`/`Critical`, confidences
between 0 and 1, and `default.yaml` must define every setting.

Check what a process actually loaded:

```bash
uv run python -c "from common.config import DEFAULT as c; print(c.hospital_config_source, c.dispatch_mode, c.outbound_min_criticality)"
```

### Vital trends: when is a vital deteriorating?

Each event carries a history of 10–30 readings per vital (HR, RespRate, SpO2, Systolic, Diastolic,
Temp). `common/vitals_trends.py` classifies each one against the `vitals_trends:` section of the
[hospital config](#per-hospital-configuration). A vital is **deteriorating** only when all three hold:

1. **Consistent:** the vendored Mann-Kendall test gives p < `alpha` (default 0.05).
2. **Big enough:** the change over the window is at least the vital's `min_change`. The change is
   Sen's slope per *second* times the window span, so unevenly spaced readings are handled.
3. **Abnormal and getting worse:** the latest reading is outside the vital's `normal` range and moving
   further away from it. A latest reading inside `normal` is never deteriorating.

Moving back toward normal by at least `min_change` is **improving**. Everything else is **stable**, and
the event records why:

| `reason` | Meaning |
|---|---|
| `not_significant` | no consistent drift (p ≥ alpha) |
| `below_min_change` | consistent, but smaller than `min_change` |
| `within_normal` | big enough, but the latest reading is inside the normal range |
| `toward_normal` | improving |
| `away_from_normal` | deteriorating |

**Defaults** (`config/hospitals/default.yaml`, `vitals_trends:`). Normal ranges are the MEWS 0-score bands; the
minimum changes are POC starting values pending clinical sign-off.

| Vital | `min_change` | `normal` |
|---|---|---|
| HR | 10 bpm | 51–100 |
| RespRate | 4 /min | 9–14 |
| SpO2 | 3 % | 94–100 |
| Systolic | 20 mmHg | 101–200 |
| Diastolic | 10 mmHg | 60–90 (MEWS does not score it; conventional range) |
| Temp | 1.0 °F | 95.0–101.1 |

**Per hospital.** Override any of these in `config/hospitals/<HOSPITAL_ID>.yaml` under
`vitals_trends:`, with only the keys that differ (see
[Per-hospital configuration](#per-hospital-configuration)). An invalid file stops startup. Policy is
applied when an event is analysed (`cli.ingest`), so a change affects **new events only**: stored
events keep the verdict they were given.

**Reading it in the app.** The event panel lists *deteriorating* vitals and, separately, *trends, not
alerting*, each with the change, the time span, the current value and the threshold or range it was
judged against. The p-value is only a tooltip: it says how *consistent* the drift is, not how large.
With 25 readings, a steady 1–2 breaths/min drift scores p < 0.001.

**Level vs trend.** MEWS scores each vital's *current level*; the trend judges its *change*. They can
disagree. RR at a steady 23–25 /min scores 2 MEWS points (shown as `Respiratory Rate 25 → 2`, and it
can make MEWS reach the threshold), while its trend is `below_min_change`. Both are correct.

**Simulator caveat.** The simulated Diastolic BP is 35–49 mmHg for nearly every patient, below the 60–90
normal range, so any 10 mmHg drift in it is flagged. That is an artefact of the data, not of the
policy, and it is deliberately left as is. To quiet it for a demo, lower the Diastolic `normal` floor
in the hospital file, for example `vitals_trends: {vitals: {Diastolic: {normal: [30, 90]}}}`.
