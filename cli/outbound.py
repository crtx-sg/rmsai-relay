"""Phase 7 full loop: drop an HDF5 file -> detect -> persist -> call out -> follow-up -> ack.

  python -m cli.outbound --file data/fixtures/PT1155_2026-06.h5 \
      --follow-up "what were the vitals at the event" --ack "yes I acknowledge"

For each event in the file: run the Phase 1 pipeline -> DeviceEvent, persist it (Phase 4
event flow), and if it clears the criticality gate, place a (simulated) outbound call, speak the
report, answer grounded follow-ups, and record the acknowledgment. Real telephony (LiveKit SIP)
is a deployment step; this uses a SimulatedCaller so the loop is demonstrable end-to-end.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

from common.bed_assignment import BedAssignmentStub
from common.config import DEFAULT, Config
from common.deid import get_deidentifier
from common.patient_history import PatientHistoryStub
from common.providers import DeidentifyingLLM, get_llm_provider
from inference.ecg_model import get_ecg_model
from inference.pipeline import process_window
from inference.vitals_analysis import MewsVitalsAnalysis
from ingest.hdf5_reader import read_hdf5_file
from kb.graph.driver import GraphDriver
from kb.graph.ingest import ingest_patient_record
from kb.graph.schema import migrate
from kb.hybrid.retriever import HybridRetriever
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore
from memory.episodic import EpisodicMemory
from memory.working import WorkingMemory
from common.notify import SimulatedSmsNotifier
from orchestrator.event_flow import process_device_event
from orchestrator.orchestrator import Orchestrator
from orchestrator.outbound_flow import run_outbound, run_text_notify, should_call
from voice.outbound import CallOutcome, SimulatedCaller, get_caller


def _ensure_patient(driver, beds: BedAssignmentStub, patient_id: str) -> tuple[str, str]:
    """Auto-create an unknown patient (G8): fetch synthetic history + assign a bed, then ingest."""
    rows = driver.run_read("MATCH (p:Patient {id:$id}) RETURN p.id AS id", id=patient_id)
    unit, bed = beds.assign(patient_id)
    if not rows:
        history = PatientHistoryStub().get(patient_id).to_dict()
        ingest_patient_record(driver, history, bed=(unit, bed))
    return unit, bed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True)
    parser.add_argument("--channel", choices=["voice", "text"], default="voice",
                        help="alert by outbound voice call or by text message")
    parser.add_argument("--follow-up", action="append", default=[], dest="follow_ups")
    parser.add_argument("--ack", default="yes I acknowledge")
    parser.add_argument("--caller", choices=["simulated", "livekit"], default="simulated",
                        help="voice channel: 'simulated' (offline) or 'livekit' (real SIP via LiveKit Cloud)")
    parser.add_argument("--no-answer", action="store_true", help="simulate the call not answering")
    parser.add_argument("--fail-delivery", action="store_true", help="simulate text delivery failure")
    parser.add_argument("--notifier", choices=["simulated", "twilio"], default="simulated",
                        help="text channel: 'simulated' prints the message; 'twilio' sends real SMS")
    parser.add_argument("--number", default="+15551234567")
    parser.add_argument("--min-criticality", default="High")
    parser.add_argument("--embedder", default="hashing", choices=["auto", "bge", "hashing"])
    args = parser.parse_args(argv)

    config = replace(
        Config.from_env(), outbound_enabled=True, outbound_call_number=args.number,
        outbound_min_criticality=args.min_criticality,
    )
    utterances = [*args.follow_ups, args.ack, "yes"]  # follow-ups, then ack + confirm-back

    model = get_ecg_model()  # stub until real weights present
    vitals = MewsVitalsAnalysis()
    beds = BedAssignmentStub()

    vector = VectorRetriever.build(
        store=QdrantStore.connect(DEFAULT.qdrant_url, "rmsai_docs"), embedder_name=args.embedder
    )
    vector.index_dir("docs")
    driver = GraphDriver.from_config(DEFAULT)
    migrate(driver)
    orch = Orchestrator(
        working=WorkingMemory.from_config(), hybrid=HybridRetriever(vector, driver),
        episodic=EpisodicMemory.from_config(embedder_name=args.embedder),
        llm=DeidentifyingLLM(get_llm_provider("echo"), get_deidentifier("regex")), driver=driver,
    )
    if args.channel == "voice" and args.caller == "livekit":
        caller = get_caller("livekit", config)  # real SIP via LiveKit Cloud
    else:
        caller = SimulatedCaller(
            [CallOutcome.NO_ANSWER] * 5 if args.no_answer else [CallOutcome.ANSWERED]
        )
    if args.channel == "text" and args.notifier == "twilio":
        from common.notify import get_notifier  # noqa: PLC0415

        notifier = get_notifier(
            "twilio",
            account_sid=os.environ["TWILIO_ACCOUNT_SID"],
            auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            from_number=os.environ.get("OUTBOUND_FROM", DEFAULT.outbound_from),
        )
    else:
        notifier = SimulatedSmsNotifier(deliver=not args.fail_delivery)

    try:
        for event in read_hdf5_file(args.file):
            de = process_window(event, model, vitals)
            unit, bed = _ensure_patient(driver, beds, de.window.patient_ref)
            process_device_event(de, driver, vector, bed=(unit, bed))
            call, reason = should_call(de, config)
            tag = f"{de.window.patient_ref}/{de.event_type} (conf {de.confidence:.2f})"
            if not call:
                print(f"[skip] {tag} @ {bed}: {reason}")
                continue
            if args.channel == "text":
                result = run_text_notify(
                    de, driver=driver, orchestrator=orch, notifier=notifier, to=args.number,
                    utterances=utterances, config=config, bed=bed,
                )
                label, verb = "TEXT", "sms"
            else:
                result = run_outbound(
                    de, driver=driver, orchestrator=orch, caller=caller, utterances=utterances,
                    config=config, bed=bed,
                )
                label, verb = "CALL", "agent"
            print(f"[{label}] {tag} @ {bed} -> {result.outcome} (attempts {result.attempts})")
            for line in result.transcript:
                print(f"    {verb}: {line}")
            print(f"    => status: {result.status}")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
