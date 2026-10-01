# common

Frozen contracts and cross-cutting utilities. Phase 0.

- `schemas.py`, `event_types.py`, `interfaces.py`: the frozen contracts and the event-type vocabulary.
- `config.py`: every setting (`.env`-overridable), including `Config.telephony()` (the config the phone-call paths use in the telephony split) and `TELEPHONY_CARRIER` with the SignalWire/Twilio settings.
- `providers.py`: `LLMProvider`: `EchoLLM` (default), `OllamaProvider`, `DeidentifyingLLM`.
- `deid.py`: de-identification (regex / presidio). `redacting_logger.py`, `audit.py`, `tracing.py`.
- `notify.py`: SMS: `SimulatedSmsNotifier`, `TwilioSmsNotifier` (REST over stdlib HTTP), `notifier_from_env`.
- `criticality.py`, `protocol_loader.py` + `protocols/`, `ecg_model_stub.py`, `bed_assignment.py`, `patient_history.py`, `preflight.py`.
