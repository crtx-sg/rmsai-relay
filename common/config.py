"""POC configuration defaults (all overridable via environment / .env).

Every value here is a deliberate POC simplification to revisit for production (spec §13).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Load `.env` from the repo root into the process env (does NOT override existing vars).

    Minimal, dependency-free. Lets `.env` configure the Python app the same way it configures
    docker-compose. Exported shell vars take precedence over `.env`.
    """
    if os.environ.get("RMSAI_NO_DOTENV"):  # tests/CI set this for a hermetic env
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.split(" #", 1)[0].strip().strip('"').strip("'")  # drop inline comment + quotes
        os.environ.setdefault(key.strip(), val)


# Clinical vocabulary that biases Whisper STT (G15) — helps small models (e.g. tiny.en) recognise
# arrhythmia/drug/vitals terms and the acknowledgement words.
CLINICAL_STT_PROMPT = (
    "Hey Vios. Vios. "  # wake word — prime Whisper so it mis-hears the brand word less often
    "Arrhythmia, atrial fibrillation, ventricular tachycardia, ventricular fibrillation, "
    "bradycardia, tachycardia, SVT, PVC, AV block, ST elevation, MEWS, SpO2, systolic, diastolic, "
    "beta-blocker, anticoagulant, amiodarone, defibrillation, cardioversion, escalate, acknowledge, "
    "bed, unit, criticality."
)


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _b(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    # Confidence thresholds (G7)
    fp_suppress_min_confidence: float = 0.80  # suppress as FP only above this; below ⇒ uncertain
    low_confidence_caveat: float = 0.60  # top-class < this ⇒ low_confidence + caveat

    # Inbound auth (G10)
    inbound_auth_pin: str = "1234"

    # Criticality escalation (G1) — gates the outbound call + protocol matching. All configurable:
    # any event other than the normal baseline, a MEWS score at/above the threshold, or a
    # deteriorating vital trend escalates criticality to at least High.
    criticality_normal_event: str = "NORMAL_SINUS"  # the only event treated as non-critical
    criticality_mews_threshold: int = 3  # MEWS score at/above this ⇒ escalate to High
    criticality_escalate_on_deteriorating: bool = True  # any deteriorating vital ⇒ escalate to High
    # Call even when the ECG is a (confident) false positive, if the patient's vitals warrant it
    # (MEWS >= threshold or deteriorating). Overrides the NORMAL_SINUS ⇒ no-call guard (spec D10).
    criticality_fp_override_on_vitals: bool = True

    # Outbound calling (§6.1 / D16)
    outbound_enabled: bool = False
    outbound_call_number: str = ""
    outbound_from: str = ""
    outbound_min_criticality: str = "High"
    # Arrhythmia confidence gate: a non-normal prediction is only asserted AS A RHYTHM when the
    # model's confidence is at/above this. Vitals never raise the bar (they cannot make an uncertain
    # classification true) — below it the alert is either withheld (calm vitals; event still
    # persisted) or re-based as a vitals-driven alert naming the vital, with the rhythm marked
    # unconfirmed. See common.criticality.alert_basis.
    outbound_min_arrhythmia_confidence: float = 0.60
    outbound_max_retries: int = 2
    outbound_retry_delay_s: int = 30

    # Dispatch surface for a critical event (Phase 9). The criticality gate is unchanged; this
    # only chooses where a critical event is delivered:
    #   app       -> push a notification into the per-hospital inbox room only
    #   call      -> the existing per-event SIP/voice (or text) alert only (today's behaviour)
    #   app+call  -> both fire
    dispatch_mode: str = "app+call"  # app | call | app+call
    # Facility the inbox room is scoped to. The companion app joins `rmsai-inbox-<hospital_id>`.
    # POC: a single configured facility; production fan-out to an on-call clinician is out of scope.
    hospital_id: str = ""

    # Datastore endpoints (lean services)
    redis_url: str = "redis://localhost:6379/0"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "rmsai_dev_pw"
    qdrant_url: str = "http://localhost:6333"

    # LLM service (self-hosted default; cloud only on synthetic data)
    llm_provider: str = "echo"  # echo (deterministic, offline) | ollama
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2"

    # Embeddings (semantic + episodic memory, vector RAG)
    embedder: str = "hashing"  # hashing (offline) | bge | auto
    bge_model: str = "BAAI/bge-small-en-v1.5"
    # Relevance gate: answer when the best passage's *semantic* similarity clears this, even if it
    # shares few words with the question. Calibrated on the committed corpus with BGE, where
    # on-topic queries scored 0.688-0.827 and off-topic 0.408-0.551. The hashing embedder's scores
    # do NOT separate the two (0.16-0.41 vs 0.11-0.42, indistinguishable), so this arm of the gate
    # only does useful work with EMBEDDER=bge; the lexical arm below carries the hashing path.
    kb_min_relevance: float = 0.60
    # When the operational regexes miss a question that is plainly about a bed/patient/event, ask
    # the LLM which graph template was meant (kb/graph/llm_router.py). Costs one extra model call,
    # but only on a miss AND only when the question looks operational — document questions never
    # trigger it. Off by default: it puts a model call in front of a Cypher lookup, which is a
    # deliberate choice to make rather than inherit.
    kb_llm_router: bool = False
    # Managed upload folder. `cli.kb_upload` copies each uploaded document here, and a corpus
    # rebuild (`cli.kb_vector index --reset`) re-indexes it alongside docs/ — so an uploaded SOP
    # survives the rebuild that would otherwise silently delete it (uploads live only in the vector
    # store, and --reset recreates the collection from disk).
    kb_upload_dir: str = "data/kb_uploads"

    # De-identification backend (before any model call)
    deid_backend: str = "regex"  # regex (offline) | presidio | auto
    deid_spacy_model: str = "en_core_web_lg"  # spaCy NER model for presidio (or en_core_web_sm)

    # Voice STT/TTS (real audio path)
    stt_backend: str = "stub"  # stub | whisper | elevenlabs
    tts_backend: str = "stub"  # stub | piper | elevenlabs
    whisper_model: str = "base.en"
    stt_language: str = "en"  # force STT language (ISO 639-1); "" / "auto" = auto-detect
    stt_initial_prompt: str = CLINICAL_STT_PROMPT  # Whisper vocab biasing (G15)
    piper_voice_path: str = ""  # path to a Piper .onnx voice
    # ElevenLabs (cloud STT "Scribe" + TTS) — for accuracy/latency benchmarking on SYNTHETIC data
    # only. Sends audio/text to a third party, so it must NEVER carry real PHI (hard rules #4/#5).
    elevenlabs_api_key: str = ""
    # Default premade voice usable on the free tier ("Adam"). NOTE: ElevenLabs moves voices behind
    # paid plans over time (e.g. "Rachel"/library voices now 402 `paid_plan_required`); set
    # ELEVENLABS_VOICE_ID to a premade voice your plan allows.
    elevenlabs_voice_id: str = "pNInz6obpgDQGcFmaJgB"
    elevenlabs_tts_model: str = "eleven_flash_v2_5"  # low-latency TTS model
    elevenlabs_stt_model: str = "scribe_v1"
    elevenlabs_tts_sample_rate: int = 22050  # PCM rate; ElevenLabs supports 16000/22050/24000/44100
    # LiveKit (self-hosted ws://localhost:7880 or LiveKit Cloud wss://<project>.livekit.cloud)
    livekit_url: str = "ws://localhost:7880"
    # Browser-facing LiveKit URL handed to the companion app by `POST /session`. Differs from
    # `livekit_url` only when the app runs behind a public edge and reaches LiveKit via a public
    # ingress (Phase 9 deploy) — the server API keeps using `livekit_url`. Empty ⇒ same as `livekit_url`.
    livekit_public_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_sip_trunk_id: str = ""  # outbound SIP trunk id (LiveKit Cloud Telephony)
    # Room each on-demand phone call gets (one per call, `<prefix><call-id>`). Shares the prefix the
    # inbound dispatch rule routes to (voice/gateway/sip-inbound.example.yaml), so both legs of the
    # phone pipeline land in the same shape of room and the worker treats them identically.
    call_room_prefix: str = "rmsai-call-"
    # Call safety rails, both passed to CreateSIPParticipantRequest. Ringing timeout bounds how long
    # we hold a trunk channel on an unanswered call; max duration is the backstop against a call that
    # is answered by voicemail and then billed until someone notices.
    sip_ringing_timeout_s: int = 30
    sip_max_call_duration_s: int = 600
    livekit_sip_room: str = "rmsai-outbound"
    # Named-agent for EXPLICIT dispatch. The worker registers under this name and no longer auto-joins
    # new rooms; instead every room that needs the agent (inbox on /session, outbound per event,
    # inbound KB demo) requests it explicitly via create_agent_dispatch. Removes the "worker must
    # start before the app / restart re-dispatch" fragility of automatic dispatch.
    livekit_agent_name: str = "rmsai-agent"
    # On worker startup, re-dispatch the agent into live `rmsai-inbox-*` rooms that lost their agent
    # (e.g. after a worker restart) so the companion app doesn't need a re-login to re-wire chat/voice.
    # Runs in a background thread; idempotent (skips rooms that already have an agent). See cli.dispatch.
    livekit_redispatch_on_start: bool = True
    # livekit-agents runs a health-check HTTP server; 8081 is its own default. It is a real bind, so
    # two workers on the same network namespace collide (`[errno 98] address already in use`) — which
    # is exactly what happens when the containerized worker uses host networking while one is also
    # running on the host. Give the container a different port instead of stopping one of them.
    livekit_worker_http_port: int = 8081
    # Wake word: after the alert, follow-up *audio* Q&A must start with this phrase (so room noise
    # and Whisper hallucinations don't trigger replies). The agent stays "awake" for the window
    # after each wake word so follow-ups don't repeat it. Text-chat turns are never gated.
    audio_wake_word: str = "hey vios"
    audio_wake_window_s: float = 30.0
    # Require the wake word to open a follow-up audio turn (post-alert Q&A on SIP/playground). On by
    # default; set false to answer every authenticated audio turn (like push-to-talk) — the escape
    # hatch when STT mishears the out-of-vocab brand word. The companion app already bypasses this
    # (the app controls the mic), so this only affects the outbound/inbound voice flows.
    audio_wake_required: bool = True
    # Companion app (inbox room): on selecting a worklist event, speak that event's stored report
    # summary aloud (in addition to scoping chat). Especially wanted for the SIP phone flow. The
    # spoken text is the Report node's `summary` (no model call); disable to keep selection silent.
    inbox_speak_on_select: bool = True
    # Episodic recall: when on, free-text answers are conditioned on recalled past Q&A ("Relevant
    # past interactions"). Off by default — keeps answers grounded only in the live KB + this
    # conversation, and avoids a small model parroting stale recalled text.
    episodic_recall: bool = False

    # Audit log
    audit_log_path: str = "data/audit.jsonl"

    # Report archive — full event-report markdown is materialized here (gitignored, like the audit
    # log). The graph `Report.uri` points at the written file; the vector index is the search copy.
    report_dir: str = "data/reports"

    # ECG strip plots — the producer renders an event's ECG lead to `{plot_dir}/{event_id}.png`
    # (gitignored) and stores the path in `MonitoredEvent.ecg_plot_ref`. Off keeps the pipeline lean.
    ecg_plot_enabled: bool = True
    plot_dir: str = "data/plots"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            fp_suppress_min_confidence=_f("FP_SUPPRESS_MIN_CONFIDENCE", 0.80),
            low_confidence_caveat=_f("LOW_CONFIDENCE_CAVEAT", 0.60),
            inbound_auth_pin=os.environ.get("INBOUND_AUTH_PIN", "1234"),
            criticality_normal_event=os.environ.get("CRITICALITY_NORMAL_EVENT", "NORMAL_SINUS"),
            criticality_mews_threshold=_i("CRITICALITY_MEWS_THRESHOLD", 3),
            criticality_escalate_on_deteriorating=_b("CRITICALITY_ESCALATE_ON_DETERIORATING", True),
            criticality_fp_override_on_vitals=_b("CRITICALITY_FP_OVERRIDE_ON_VITALS", True),
            outbound_enabled=_b("OUTBOUND_ENABLED", False),
            outbound_call_number=os.environ.get("OUTBOUND_CALL_NUMBER", ""),
            outbound_from=os.environ.get("OUTBOUND_FROM", ""),
            outbound_min_criticality=os.environ.get("OUTBOUND_MIN_CRITICALITY", "High"),
            outbound_min_arrhythmia_confidence=_f("OUTBOUND_MIN_ARRHYTHMIA_CONFIDENCE", 0.60),
            outbound_max_retries=_i("OUTBOUND_MAX_RETRIES", 2),
            outbound_retry_delay_s=_i("OUTBOUND_RETRY_DELAY_S", 30),
            dispatch_mode=os.environ.get("DISPATCH_MODE", "app+call"),
            hospital_id=os.environ.get("HOSPITAL_ID", ""),
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_password=os.environ.get("NEO4J_PASSWORD", "rmsai_dev_pw"),
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            llm_provider=os.environ.get("LLM_PROVIDER", "echo"),
            ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            llm_model=os.environ.get("LLM_MODEL", "llama3.2"),
            embedder=os.environ.get("EMBEDDER", "hashing"),
            bge_model=os.environ.get("BGE_MODEL", "BAAI/bge-small-en-v1.5"),
            kb_min_relevance=_f("KB_MIN_RELEVANCE", 0.60),
            kb_llm_router=_b("KB_LLM_ROUTER", False),
            kb_upload_dir=os.environ.get("KB_UPLOAD_DIR", "data/kb_uploads"),
            deid_backend=os.environ.get("DEID_BACKEND", "regex"),
            deid_spacy_model=os.environ.get("DEID_SPACY_MODEL", "en_core_web_lg"),
            stt_backend=os.environ.get("STT_BACKEND", "stub"),
            tts_backend=os.environ.get("TTS_BACKEND", "stub"),
            whisper_model=os.environ.get("WHISPER_MODEL", "base.en"),
            stt_language=os.environ.get("STT_LANGUAGE", "en"),
            stt_initial_prompt=os.environ.get("STT_INITIAL_PROMPT", CLINICAL_STT_PROMPT),
            piper_voice_path=os.environ.get("PIPER_VOICE_PATH", ""),
            elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY", ""),
            elevenlabs_voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            elevenlabs_tts_model=os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5"),
            elevenlabs_stt_model=os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1"),
            elevenlabs_tts_sample_rate=_i("ELEVENLABS_TTS_SAMPLE_RATE", 22050),
            livekit_url=os.environ.get("LIVEKIT_URL", "ws://localhost:7880"),
            livekit_public_url=os.environ.get("LIVEKIT_PUBLIC_URL", ""),
            livekit_api_key=os.environ.get("LIVEKIT_API_KEY", ""),
            livekit_api_secret=os.environ.get("LIVEKIT_API_SECRET", ""),
            livekit_sip_trunk_id=os.environ.get("LIVEKIT_SIP_TRUNK_ID", ""),
            call_room_prefix=os.environ.get("LIVEKIT_CALL_ROOM_PREFIX", "rmsai-call-"),
            sip_ringing_timeout_s=_i("SIP_RINGING_TIMEOUT_S", 30),
            sip_max_call_duration_s=_i("SIP_MAX_CALL_DURATION_S", 600),
            livekit_sip_room=os.environ.get("LIVEKIT_SIP_ROOM", "rmsai-outbound"),
            livekit_agent_name=os.environ.get("LIVEKIT_AGENT_NAME", "rmsai-agent"),
            livekit_redispatch_on_start=_b("LIVEKIT_REDISPATCH_ON_START", True),
            livekit_worker_http_port=_i("LIVEKIT_WORKER_HTTP_PORT", 8081),
            audio_wake_word=os.environ.get("AUDIO_WAKE_WORD", "hey vios"),
            audio_wake_window_s=_f("AUDIO_WAKE_WINDOW_S", 30.0),
            audio_wake_required=_b("AUDIO_WAKE_REQUIRED", True),
            inbox_speak_on_select=_b("INBOX_SPEAK_ON_SELECT", True),
            episodic_recall=_b("EPISODIC_RECALL", False),
            audit_log_path=os.environ.get("AUDIT_LOG_PATH", "data/audit.jsonl"),
            report_dir=os.environ.get("REPORT_DIR", "data/reports"),
            ecg_plot_enabled=_b("ECG_PLOT_ENABLED", True),
            plot_dir=os.environ.get("PLOT_DIR", "data/plots"),
        )


#: Process-wide default, populated from `.env` + environment at import.
#: Call `Config.from_env()` for a fresh read after changing env vars.
_load_dotenv()
DEFAULT = Config.from_env()
