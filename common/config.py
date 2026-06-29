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
    outbound_max_retries: int = 2
    outbound_retry_delay_s: int = 30

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
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_sip_trunk_id: str = ""  # outbound SIP trunk id (LiveKit Cloud Telephony)
    livekit_sip_room: str = "rmsai-outbound"
    # Wake word: after the alert, follow-up *audio* Q&A must start with this phrase (so room noise
    # and Whisper hallucinations don't trigger replies). The agent stays "awake" for the window
    # after each wake word so follow-ups don't repeat it. Text-chat turns are never gated.
    audio_wake_word: str = "hey vios"
    audio_wake_window_s: float = 30.0
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
            outbound_max_retries=_i("OUTBOUND_MAX_RETRIES", 2),
            outbound_retry_delay_s=_i("OUTBOUND_RETRY_DELAY_S", 30),
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
            livekit_api_key=os.environ.get("LIVEKIT_API_KEY", ""),
            livekit_api_secret=os.environ.get("LIVEKIT_API_SECRET", ""),
            livekit_sip_trunk_id=os.environ.get("LIVEKIT_SIP_TRUNK_ID", ""),
            livekit_sip_room=os.environ.get("LIVEKIT_SIP_ROOM", "rmsai-outbound"),
            audio_wake_word=os.environ.get("AUDIO_WAKE_WORD", "hey vios"),
            audio_wake_window_s=_f("AUDIO_WAKE_WINDOW_S", 30.0),
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
