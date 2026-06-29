"""Event-bus consumer CLI (§3.2): drain the Redis Stream and run the relay end-to-end.

  # producer (elsewhere): publish classified events to the bus
  python -m cli.ingest --file <f>.h5 --checkpoint <model>.pt --emit bus --stream rmsai.events

  # consumer (this CLI): persist + criticality-gated outbound, per message
  python -m cli.consume --stream rmsai.events \
      --follow-up "what were the vitals at the event" --ack "yes I acknowledge"

Reads with a Redis consumer group (so multiple consumers share the stream and messages are
acked exactly once). A new group starts at the beginning of the stream, so an existing backlog
is processed on first run. The consumer is model-free — classification happened in the producer.
"""

from __future__ import annotations

import argparse
import json
import os

from common.audit import AuditLog
from common.bed_assignment import BedAssignmentStub
from common.config import DEFAULT, Config
from common.deid import get_deidentifier
from common.notify import SimulatedSmsNotifier
from common.providers import DeidentifyingLLM, get_llm_provider
from dataclasses import replace

from kb.graph.driver import GraphDriver
from kb.graph.schema import migrate
from kb.hybrid.retriever import HybridRetriever
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore
from memory.episodic import EpisodicMemory
from memory.working import WorkingMemory
from orchestrator.bus_consumer import process_bus_event
from orchestrator.orchestrator import Orchestrator
from voice.outbound import CallOutcome, SimulatedCaller, get_caller


def _ensure_group(client, stream: str, group: str) -> None:
    """Create the consumer group at the stream head (id=0 → includes backlog); ignore if it exists."""
    import redis  # noqa: PLC0415

    try:
        client.xgroup_create(stream, group, id="0", mkstream=True)
    except redis.ResponseError as exc:  # noqa: PERF203
        if "BUSYGROUP" not in str(exc):
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", default="rmsai.events", help="Redis Stream to consume.")
    parser.add_argument("--group", default="rmsai.relay", help="Consumer group name.")
    parser.add_argument("--consumer", default="c1", help="Consumer name within the group.")
    parser.add_argument("--redis-url", default=DEFAULT.redis_url)
    parser.add_argument("--channel", choices=["voice", "text"], default="voice",
                        help="alert by outbound voice call or by text message")
    parser.add_argument("--follow-up", action="append", default=[], dest="follow_ups")
    parser.add_argument("--ack", default="yes I acknowledge")
    parser.add_argument("--caller", choices=["simulated", "livekit"], default="simulated")
    parser.add_argument("--transport", choices=["sip", "webrtc"], default="sip",
                        help="livekit voice: 'sip' dials a phone; 'webrtc' stages the alert + prints "
                             "a join token (test the audio loop from a browser, no phone).")
    parser.add_argument("--notifier", choices=["simulated", "twilio"], default="simulated")
    parser.add_argument("--number", default="+15551234567")
    parser.add_argument("--min-criticality", default="High")
    parser.add_argument("--embedder", default="hashing", choices=["auto", "bge", "hashing"])
    parser.add_argument("--count", type=int, default=10, help="Max messages per read batch.")
    parser.add_argument("--block-ms", type=int, default=5000, help="Block this long awaiting messages.")
    parser.add_argument("--once", action="store_true", help="Process one batch (incl. backlog) then exit.")
    parser.add_argument("--max-events", type=int, default=None, help="Exit after N messages.")
    args = parser.parse_args(argv)

    import redis  # noqa: PLC0415

    config = replace(
        Config.from_env(), outbound_enabled=True, outbound_call_number=args.number,
        outbound_min_criticality=args.min_criticality,
    )
    utterances = [*args.follow_ups, args.ack, "yes"]
    audit = AuditLog(DEFAULT.audit_log_path)
    beds = BedAssignmentStub()

    vector = VectorRetriever.build(
        store=QdrantStore.connect(DEFAULT.qdrant_url, "rmsai_docs"), embedder_name=args.embedder
    )
    # Append (don't reset): ensure the clinical docs are present without wiping event-report
    # narratives that earlier consume runs archived into the same collection.
    vector.index_dir("docs", reset=False)
    driver = GraphDriver.from_config(DEFAULT)
    migrate(driver)
    orch = Orchestrator(
        working=WorkingMemory.from_config(), hybrid=HybridRetriever(vector, driver),
        episodic=EpisodicMemory.from_config(embedder_name=args.embedder),
        llm=DeidentifyingLLM(get_llm_provider(config.llm_provider, config),
                             get_deidentifier(config.deid_backend)), driver=driver,
        episodic_recall=config.episodic_recall,
    )
    caller_factory = None
    alert_store = None
    if args.channel == "voice" and args.caller == "livekit":
        from voice.outbound import LiveKitCaller  # noqa: PLC0415
        from voice.outbound_alert import OutboundAlertStore  # noqa: PLC0415

        # Per-event room + alert hand-off; the worker (cli.voice_worker) drives the real audio loop.
        alert_store = OutboundAlertStore.from_config(config)
        if args.transport == "webrtc":
            # No phone: "answer" immediately so the alert is staged; the clinician joins over WebRTC.
            caller_factory = lambda room: SimulatedCaller([CallOutcome.ANSWERED])  # noqa: E731
        else:
            caller_factory = lambda room: LiveKitCaller(config, room=room)  # noqa: E731
        caller = None
    else:
        caller = SimulatedCaller([CallOutcome.ANSWERED] * max(args.count, 1))
    if args.channel == "text" and args.notifier == "twilio":
        from common.notify import get_notifier  # noqa: PLC0415

        notifier = get_notifier(
            "twilio", account_sid=os.environ["TWILIO_ACCOUNT_SID"],
            auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            from_number=os.environ.get("OUTBOUND_FROM", DEFAULT.outbound_from),
        )
    else:
        notifier = SimulatedSmsNotifier()

    # socket_timeout must outlive the server-side BLOCK window, else an idle blocking read
    # (no new messages) raises redis TimeoutError instead of returning nil. Give it headroom.
    client = redis.Redis.from_url(args.redis_url, socket_timeout=args.block_ms / 1000 + 5)
    _ensure_group(client, args.stream, args.group)
    print(f"[consume] group={args.group} consumer={args.consumer} stream={args.stream}")

    processed = 0
    try:
        while True:
            try:
                resp = client.xreadgroup(
                    args.group, args.consumer, {args.stream: ">"},
                    count=args.count, block=args.block_ms,
                )
            except redis.exceptions.TimeoutError:
                # Idle blocking read elapsed (or a transient slow read) with no new messages.
                # Treat as an empty batch: exit on --once, otherwise keep waiting.
                if args.once:
                    break
                continue
            if not resp:
                if args.once:
                    break
                continue
            empty_batch = True
            for _stream, entries in resp:
                for msg_id, fields in entries:
                    empty_batch = False
                    mid = msg_id.decode()
                    try:
                        payload = json.loads(fields[b"data"])
                        result = process_bus_event(
                            payload, driver=driver, vector=vector, orchestrator=orch, beds=beds,
                            utterances=utterances, channel=args.channel, caller=caller,
                            notifier=notifier, to=args.number, config=config,
                            caller_factory=caller_factory, alert_store=alert_store,
                        )
                        _report(result)
                        if args.transport == "webrtc" and result.called:
                            _print_webrtc_join(config, result.event_uuid)
                        audit.write(actor="cli.consume", action="consume_event",
                                    subject=result.patient_ref, outcome="processed",
                                    stream=args.stream, msg_id=mid)
                    except Exception as exc:  # noqa: BLE001 - poison message: ack + record, don't wedge the group
                        print(f"[poison] {mid}: {type(exc).__name__}: {exc}")
                        audit.write(actor="cli.consume", action="consume_event",
                                    subject="?", outcome="poison", stream=args.stream, msg_id=mid)
                    finally:
                        client.xack(args.stream, args.group, msg_id)
                        processed += 1
            if args.max_events is not None and processed >= args.max_events:
                break
            if args.once and not empty_batch:
                # drained this batch; loop once more to catch remaining backlog, else block returns []
                continue
    except KeyboardInterrupt:
        pass
    finally:
        driver.close()
    print(f"[consume] processed {processed} message(s)")
    return 0


def _print_webrtc_join(config, event_uuid: str) -> None:
    """Print a LiveKit join token so a clinician can answer this event's call over WebRTC."""
    from voice.livekit_cloud import access_token, is_configured  # noqa: PLC0415

    room = f"rmsai-outbound-{event_uuid}"
    if not is_configured(config):
        print(f"    [webrtc] alert staged for room {room} (set LIVEKIT_* to mint a join token)")
        return
    token = access_token(identity="clinician", room=room, name="clinician", config=config)
    print(f"    [webrtc] join room to take the call:")
    print(f"      URL:   {config.livekit_url}")
    print(f"      Room:  {room}")
    print(f"      Token: {token}")


def _report(result) -> None:
    tag = f"{result.patient_ref}/{result.event_type}"
    if not result.called:
        print(f"[skip] {tag} @ {result.bed}: {result.decision_reason}")
        return
    ob = result.outbound
    print(f"[{result.bed}] {tag} -> {ob.outcome} (attempts {ob.attempts}) status={ob.status}")
    for line in ob.transcript:
        print(f"    agent: {line}")


if __name__ == "__main__":
    raise SystemExit(main())
