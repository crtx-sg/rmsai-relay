"""Event-bus consumer (§3.2): drain `DeviceEvent`s off the Redis Stream and run the relay.

The producer (`cli.ingest --emit bus`) classifies HDF5 events and `XADD`s each enriched
`DeviceEvent` to a stream. This consumer is the other half: for each message it reconstructs the
event, persists + archives it (Phase 4 event flow), and — if it clears the criticality gate —
alerts the clinician by outbound voice call or text (Phase 7). The consumer is **model-free**:
classification already happened upstream, so no checkpoint is loaded here.

`process_bus_event` handles a single decoded payload and is fully testable with stubs; the Redis
consumer-group loop lives in `cli.consume`.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.bed_assignment import BedAssignmentStub
from common.config import DEFAULT, Config
from inference.serialize import dict_to_event
from kb.graph.driver import GraphDriver
from kb.vector.retriever import VectorRetriever
from orchestrator.event_flow import process_device_event
from orchestrator.outbound_flow import OutboundResult, run_outbound, run_text_notify, should_call
from orchestrator.patient_bootstrap import ensure_patient
from orchestrator.report import spoken_report
from voice.outbound_alert import OutboundAlert


@dataclass
class ConsumeResult:
    """What the consumer did with one message (for logging + tests)."""

    event_uuid: str
    patient_ref: str
    event_type: str
    bed: str
    persisted: bool
    called: bool
    decision_reason: str
    outbound: OutboundResult | None = None


def process_bus_event(
    payload: dict,
    *,
    driver: GraphDriver,
    vector: VectorRetriever,
    orchestrator,
    beds: BedAssignmentStub,
    utterances: list[str],
    channel: str = "voice",
    caller=None,
    notifier=None,
    to: str | None = None,
    config: Config = DEFAULT,
    caller_factory=None,
    alert_store=None,
) -> ConsumeResult:
    """Reconstruct, persist, and (if the gate clears) alert for one bus payload.

    Always persists the event (the chart records every event, false positives included); the
    criticality gate only governs whether we dial/text out. Returns a `ConsumeResult` summary.

    `caller_factory` (room -> Caller) + `alert_store` enable the **live LiveKit** voice path: a
    per-event room is created, the alert is staged in the store for the worker, and the call is
    placed without scripting the conversation (the worker drives the real STT/TTS loop). Without
    them, the voice path uses the supplied `caller` and the offline scripted loop.
    """
    event = dict_to_event(payload)
    w = event.window
    print(f"[consume] received event {w.event_id} type={event.event_type} "
          f"conf={event.confidence:.2f} patient={w.patient_ref}", flush=True)
    unit, bed = ensure_patient(driver, beds, w.patient_ref)
    bed_label = f"{unit}/{bed}"
    print(f"[consume] patient {w.patient_ref} -> bed {bed_label}", flush=True)

    process_device_event(event, driver, vector, bed=(unit, bed), config=config)
    print(f"[consume] persisted MonitoredEvent {w.event_id} -> Neo4j graph", flush=True)
    print(f"[consume] archived report narrative -> Qdrant vector store", flush=True)

    call, reason = should_call(event, config)
    if not call:
        print(f"[consume] no call: {reason} (event still persisted)", flush=True)
        return ConsumeResult(
            event_uuid=w.event_id, patient_ref=w.patient_ref, event_type=event.event_type,
            bed=bed_label, persisted=True, called=False, decision_reason=reason,
        )
    if reason.startswith("fp_override"):
        print(f"[consume] FALSE-POSITIVE OVERRIDE: ECG classified {event.event_type} "
              f"(false positive), but the patient's vitals warrant a call [{reason}]. "
              f"Calling anyway — vitals/MEWS-driven escalation, not the rhythm.", flush=True)

    to = to or config.outbound_call_number
    if channel == "text":
        result = run_text_notify(
            event, driver=driver, orchestrator=orchestrator, notifier=notifier, to=to,
            utterances=utterances, config=config, bed=bed_label,
        )
    elif caller_factory is not None:  # live LiveKit audio: stage the alert, place the call, hand off
        room = f"rmsai-outbound-{w.event_id}"
        print(f"[consume] criticality gate PASSED ({reason}); staging alert for room {room}",
              flush=True)
        alert_store.put(OutboundAlert(
            session_id=room, patient_ref=w.patient_ref, event_id=w.event_id,
            spoken_alert=spoken_report(event, bed=bed_label, config=config), bed=bed_label,
        ))
        print(f"[consume] alert staged in Redis; placing call -> worker will join room {room}",
              flush=True)
        result = run_outbound(
            event, driver=driver, orchestrator=orchestrator, caller=caller_factory(room),
            utterances=utterances, config=config, bed=bed_label, live_audio=True,
        )
    else:
        result = run_outbound(
            event, driver=driver, orchestrator=orchestrator, caller=caller,
            utterances=utterances, config=config, bed=bed_label,
        )
    return ConsumeResult(
        event_uuid=w.event_id, patient_ref=w.patient_ref, event_type=event.event_type,
        bed=bed_label, persisted=True, called=result.called, decision_reason=reason,
        outbound=result,
    )
