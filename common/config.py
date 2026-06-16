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
    stt_backend: str = "stub"  # stub | whisper
    tts_backend: str = "stub"  # stub | piper
    whisper_model: str = "base.en"
    stt_initial_prompt: str = CLINICAL_STT_PROMPT  # Whisper vocab biasing (G15)
    piper_voice_path: str = ""  # path to a Piper .onnx voice
    # LiveKit (self-hosted ws://localhost:7880 or LiveKit Cloud wss://<project>.livekit.cloud)
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_sip_trunk_id: str = ""  # outbound SIP trunk id (LiveKit Cloud Telephony)
    livekit_sip_room: str = "rmsai-outbound"

    # Audit log
    audit_log_path: str = "data/audit.jsonl"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            fp_suppress_min_confidence=_f("FP_SUPPRESS_MIN_CONFIDENCE", 0.80),
            low_confidence_caveat=_f("LOW_CONFIDENCE_CAVEAT", 0.60),
            inbound_auth_pin=os.environ.get("INBOUND_AUTH_PIN", "1234"),
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
            stt_initial_prompt=os.environ.get("STT_INITIAL_PROMPT", CLINICAL_STT_PROMPT),
            piper_voice_path=os.environ.get("PIPER_VOICE_PATH", ""),
            livekit_url=os.environ.get("LIVEKIT_URL", "ws://localhost:7880"),
            livekit_api_key=os.environ.get("LIVEKIT_API_KEY", ""),
            livekit_api_secret=os.environ.get("LIVEKIT_API_SECRET", ""),
            livekit_sip_trunk_id=os.environ.get("LIVEKIT_SIP_TRUNK_ID", ""),
            livekit_sip_room=os.environ.get("LIVEKIT_SIP_ROOM", "rmsai-outbound"),
            audit_log_path=os.environ.get("AUDIT_LOG_PATH", "data/audit.jsonl"),
        )


#: Process-wide default, populated from `.env` + environment at import.
#: Call `Config.from_env()` for a fresh read after changing env vars.
_load_dotenv()
DEFAULT = Config.from_env()
