# common

Frozen contracts and cross-cutting utilities. Phase 0.

- `schemas.py`, `event_types.py`, `interfaces.py`: the frozen contracts and the event-type vocabulary.
- `config.py`: every setting (`.env`-overridable), including `Config.telephony()` (the config the phone-call paths use in the telephony split) and `TELEPHONY_CARRIER` with the SignalWire/Twilio settings. Escalation and outbound fields default from the hospital files (below).
- `hospital_config.py`: loads `config/hospitals/default.yaml` + `<HOSPITAL_ID>.yaml` (escalation, outbound, vital trends), validates them, and maps each key to its `Config` field and environment override (`SETTINGS`).
- `vitals_trends.py`: the clinical-significance policy for vital trends (min change + normal range on top of the Mann-Kendall p-value).
- `providers.py`: `LLMProvider`: `EchoLLM` (default), `OllamaProvider`, `AnthropicProvider`, `OpenAICompatProvider` (Gemini, OpenAI and any OpenAI-compatible API), `DeidentifyingLLM` (wraps every one); `get_llm_provider` picks by `LLM_PROVIDER`.
- `deid.py`: de-identification (regex / presidio). `redacting_logger.py`, `audit.py`, `tracing.py`.
- `notify.py`: SMS: `SimulatedSmsNotifier`, `TwilioSmsNotifier` (REST over stdlib HTTP), `notifier_from_env`.
- `criticality.py`, `protocol_loader.py` + `protocols/`, `ecg_model_stub.py`, `bed_assignment.py`, `patient_history.py`, `preflight.py`.
