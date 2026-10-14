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

from common.audit import AuditLog
from common.bed_assignment import BedAssignmentStub
from common.config import DEFAULT, Config
from common.deid import get_deidentifier
from common.preflight import service_unreachable
from common.providers import DeidentifyingLLM, get_llm_provider
from dataclasses import replace

from kb.graph.driver import GraphDriver
from kb.graph.schema import migrate
from kb.hybrid.retriever import HybridRetriever
from kb.vector.retriever import VectorRetriever
from kb.vector.store import QdrantStore
from memory.episodic import EpisodicMemory
from memory.working import WorkingMemory
from orchestrator.bus_consumer import _mode_includes, process_bus_event, run_queued_call
from orchestrator.orchestrator import Orchestrator
from voice.outbound import CallOutcome, SimulatedCaller, get_caller


def _ensure_group(client, stream: str, group: str, redis_url: str) -> None:
    """Create the consumer group at the stream head (id=0 → includes backlog); ignore if it exists."""
    import redis  # noqa: PLC0415

    try:
        client.xgroup_create(stream, group, id="0", mkstream=True)
    except redis.exceptions.ConnectionError as exc:
        # First Redis call of the run — the natural place to catch "the stores aren't running" and
        # say so, rather than emitting a connection-pool traceback. `from None` drops that noise.
        # Report the URL actually used (--redis-url), not the config default.
        raise service_unreachable("Redis (the event bus)", redis_url, exc) from None
    except redis.ResponseError as exc:  # noqa: PERF203
        if "BUSYGROUP" not in str(exc):
            raise


_FAILURES = "{stream}:{group}:failures"  # hash msg_id -> failed attempts so far


def _audit(audit, **fields) -> None:
    """A consume audit line; an unwritable log must never crash the loop or change an outcome."""
    try:
        audit.write(actor="cli.consume", action="consume_event", subject="?", **fields)
    except Exception as exc:  # noqa: BLE001
        print(f"[consume] audit write failed: {type(exc).__name__}: {exc}", flush=True)


def handle_entry(client, stream: str, group: str, msg_id, fields, work, *, audit,
                 max_attempts: int = 3) -> str:
    """Process one stream entry: 'processed' | 'poison' | 'retry' | 'dead'.

    Acks only once the message is finished with: processed, undecodable (poison), or failed
    `max_attempts` times (copied to `<stream>.dead`, then acked). Any other failure (Neo4j
    restarting, a network blip) leaves it pending so `reclaim_pending` retries it. Before this,
    every failure was acked as "poison" and the alarm was silently lost.
    """
    mid = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
    try:
        payload = json.loads(fields[b"data"])
    except (KeyError, TypeError, ValueError) as exc:
        print(f"[poison] {mid}: {type(exc).__name__}: {exc}", flush=True)
        _audit(audit, outcome="poison", stream=stream, msg_id=mid)
        client.xack(stream, group, msg_id)
        return "poison"
    failures = _FAILURES.format(stream=stream, group=group)
    try:
        work(payload, mid)
    except Exception as exc:  # noqa: BLE001 - retried, then dead-lettered; never wedges the group
        n = int(client.hincrby(failures, mid, 1))
        if n < max_attempts:
            print(f"[retry] {mid}: {type(exc).__name__}: {exc} (attempt {n}/{max_attempts}, "
                  f"left pending)", flush=True)
            _audit(audit, outcome="retry", stream=stream, msg_id=mid, attempt=n)
            return "retry"
        client.xadd(f"{stream}.dead", {"data": fields[b"data"], "msg_id": mid,
                                       "error": type(exc).__name__},
                    maxlen=10_000, approximate=True)
        client.hdel(failures, mid)
        client.xack(stream, group, msg_id)
        print(f"[dead] {mid}: {type(exc).__name__}: {exc} after {n} attempts -> {stream}.dead",
              flush=True)
        _audit(audit, outcome="dead", stream=stream, msg_id=mid, attempts=n)
        return "dead"
    client.hdel(failures, mid)
    client.xack(stream, group, msg_id)
    return "processed"


def reclaim_pending(client, stream: str, group: str, consumer: str, *, min_idle_ms: int,
                    count: int) -> list:
    """Claim messages pending longer than `min_idle_ms` (failed here, or orphaned by a crashed
    consumer) for this consumer, and return them for another try."""
    resp = client.xautoclaim(stream, group, consumer, min_idle_time=min_idle_ms,
                             start_id="0-0", count=count)
    return [(mid, fields) for mid, fields in resp[1] if fields]


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
    parser.add_argument("--notifier", choices=["simulated", "signalwire"], default="simulated",
                        help="SMS backend: text-channel delivery, and the voice channel's fallback "
                             "when a call goes unanswered. 'simulated' sends nothing (logged and "
                             "recorded as sms_simulated); 'signalwire' sends real SMS "
                             "from the hospital's outbound.from number")
    parser.add_argument("--number", default=None,
                        help="destination (E.164); default OUTBOUND_CALL_NUMBER. A placeholder is used "
                             "only for simulated runs with neither set")
    parser.add_argument("--min-criticality", default="High")
    parser.add_argument("--embedder", default=DEFAULT.embedder, choices=["auto", "bge", "hashing"],
                        help="Must match the embedder the KB collections were built with "
                             f"(default from EMBEDDER: {DEFAULT.embedder}).")
    parser.add_argument("--count", type=int, default=10, help="Max messages per read batch.")
    parser.add_argument("--block-ms", type=int, default=5000, help="Block this long awaiting messages.")
    parser.add_argument("--once", action="store_true", help="Process one batch (incl. backlog) then exit.")
    parser.add_argument("--trend-samples", action="store_true",
                        help="Append to each [event] line the vital readings behind each "
                             "deteriorating trend (what 'rising'/'falling' was computed from).")
    parser.add_argument("--perf-every", type=int, default=10,
                        help="Print a model-performance summary every N labelled events (0 = only "
                             "on exit). See also `cli.model_perf` for the summary from the graph.")
    parser.add_argument("--max-events", type=int, default=None, help="Exit after N messages.")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Tries per message before it is dead-lettered to <stream>.dead.")
    parser.add_argument("--retry-idle-ms", type=int, default=600_000,
                        help="A failed (or orphaned) message is retried after it sat this long. "
                             "Longer than any call: a message still being worked on (a blocking "
                             "call) must not be taken over by a second consumer.")
    args = parser.parse_args(argv)

    from voice.outbound import resolve_destination  # noqa: PLC0415

    real = args.caller == "livekit" or args.notifier != "simulated"
    try:
        destination = resolve_destination(args.number, Config.from_env().outbound_call_number,
                                          real=real)
    except ValueError as exc:
        parser.error(str(exc))

    import redis  # noqa: PLC0415

    config = replace(
        Config.from_env(), outbound_enabled=True, outbound_call_number=destination,
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
    dispatch_fn = None
    alert_store = None
    call_end_waiter = None
    use_call_queue = False
    if args.channel == "voice" and args.caller == "livekit":
        from voice.outbound_alert import OutboundAlertStore  # noqa: PLC0415

        # Per-event room + alert hand-off; the worker (cli.voice_worker) drives the real audio loop.
        # The alert is staged in the shared Redis, so it reaches the worker whichever LiveKit hosts
        # the room.
        alert_store = OutboundAlertStore.from_config(config)
        caller_factory, dispatch_fn = livekit_voice_wiring(config, args.transport)
        if args.transport == "sip":
            # One phone call at a time: the next alert waits for this one to be hung up.
            from voice.livekit_cloud import wait_until_call_ends  # noqa: PLC0415

            tel = config.telephony()
            call_end_waiter = lambda room: wait_until_call_ends(room, config=tel)  # noqa: E731
            # ...but the app must not wait for it: events are pushed as they arrive and the calls
            # queue (highest criticality first) for a single dispatcher thread.
            use_call_queue = True
        caller = None
    else:
        caller = SimulatedCaller([CallOutcome.ANSWERED] * max(args.count, 1))
    # SMS: the text channel's delivery, and the voice channel's fallback when a call goes unanswered.
    from common.notify import notifier_from_env  # noqa: PLC0415

    try:
        notifier = notifier_from_env(args.notifier, config=config)
    except ValueError as exc:
        parser.error(str(exc))

    # Phase 9 app surface: when DISPATCH_MODE includes `app`, push a worklist notification (with
    # scoped artifact links) into the hospital inbox room per critical event. The token store is
    # always minted for `app` modes; the publisher needs LiveKit configured.
    inbox_publisher = None
    token_store = None
    if _mode_includes(config.dispatch_mode, "app"):
        from live.artifact_tokens import ArtifactTokenStore  # noqa: PLC0415
        from live.inbox import InboxPublisher, inbox_room  # noqa: PLC0415
        from voice.livekit_cloud import is_configured  # noqa: PLC0415

        token_store = ArtifactTokenStore.from_config(config)
        if is_configured(config):
            inbox_publisher = InboxPublisher.from_config(config)
            print(f"[consume] dispatch={config.dispatch_mode}; inbox room {inbox_room(config)}")
        else:
            print(f"[consume] dispatch={config.dispatch_mode} includes app but LiveKit is not "
                  f"configured; inbox push disabled (set LIVEKIT_*)")

    # socket_timeout must outlive the server-side BLOCK window, else an idle blocking read
    # (no new messages) raises redis TimeoutError instead of returning nil. Give it headroom.
    client = redis.Redis.from_url(args.redis_url, socket_timeout=args.block_ms / 1000 + 5)
    _ensure_group(client, args.stream, args.group, args.redis_url)
    print(f"[consume] group={args.group} consumer={args.consumer} stream={args.stream}")

    call_queue = dispatcher = None
    if use_call_queue:
        from orchestrator.call_queue import CallDispatcher, CallQueue  # noqa: PLC0415

        call_queue = CallQueue(redis.Redis.from_url(args.redis_url))
        recovered = call_queue.recover()
        dispatcher = CallDispatcher(call_queue, lambda job: _report_call(run_queued_call(
            job, driver=driver, orchestrator=orch, utterances=utterances, config=config,
            notifier=notifier, caller_factory=caller_factory, dispatch_fn=dispatch_fn,
            alert_store=alert_store, call_end_waiter=call_end_waiter,
        )))
        dispatcher.start()
        print(f"[consume] call queue on (severity order, one call at a time); "
              f"{call_queue.pending()} waiting" + (f", {recovered} recovered from an earlier run"
                                                    if recovered else ""), flush=True)

    from orchestrator.event_log import PerfTracker  # noqa: PLC0415

    tracker = PerfTracker(every=args.perf_every, emit=lambda text: print(text, flush=True))
    processed = 0

    def _work(payload: dict, mid: str) -> None:
        result = process_bus_event(
            payload, driver=driver, vector=vector, orchestrator=orch, beds=beds,
            utterances=utterances, channel=args.channel, caller=caller,
            notifier=notifier, to=destination, config=config,
            caller_factory=caller_factory, dispatch_fn=dispatch_fn,
            alert_store=alert_store,
            inbox_publisher=inbox_publisher, token_store=token_store,
            call_end_waiter=call_end_waiter, call_queue=call_queue,
        )
        # The event is persisted and alerted by now. Everything below is reporting: a failure here
        # must not make the message "failed", or the retry would push and call again.
        try:
            if result.call_queued and dispatcher is not None:
                dispatcher.notify()
            _report(result, samples=args.trend_samples)
            if result.trace is not None:
                tracker.add(result.trace)
            if args.transport == "webrtc" and result.called:
                _print_webrtc_join(config, result.event_uuid)
            audit.write(actor="cli.consume", action="consume_event",
                        subject=result.patient_ref, outcome="processed",
                        stream=args.stream, msg_id=mid)
        except Exception as exc:  # noqa: BLE001 - reporting only; the alert already went out
            print(f"[consume] reporting failed for {mid}: {type(exc).__name__}: {exc}", flush=True)

    def _handle(entries) -> None:
        nonlocal processed
        for msg_id, fields in entries:
            outcome = handle_entry(client, args.stream, args.group, msg_id, fields, _work,
                                   audit=audit, max_attempts=args.max_attempts)
            if outcome != "retry":  # --max-events counts messages finished with, not attempts
                processed += 1

    try:
        while True:
            # Retry what failed earlier, and take over messages a crashed consumer left pending.
            _handle(reclaim_pending(client, args.stream, args.group, args.consumer,
                                    min_idle_ms=args.retry_idle_ms, count=args.count))
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
                if entries:
                    empty_batch = False
                _handle(entries)
            if args.max_events is not None and processed >= args.max_events:
                break
            if args.once and not empty_batch:
                # drained this batch; loop once more to catch remaining backlog, else block returns []
                continue
        if dispatcher is not None and call_queue.pending():
            # --once / --max-events: the queued calls are part of this run; place them first.
            print(f"[consume] waiting for {call_queue.pending()} queued call(s) before exiting",
                  flush=True)
        if dispatcher is not None:
            dispatcher.stop(drain=True)
            dispatcher = None
    except KeyboardInterrupt:
        pass
    finally:
        if dispatcher is not None:  # interrupted: unplaced calls stay queued for the next run
            dispatcher.stop()
        tracker.flush(header="on exit")
        driver.close()
    print(f"[consume] processed {processed} message(s)")
    return 0


def livekit_voice_wiring(config, transport: str):
    """`(caller_factory, dispatch_fn)` for the LiveKit voice channel, on the right LiveKit.

    `sip` rings a phone, so both the dial and the agent dispatch go to `config.telephony()`: the
    LiveKit Cloud project in split mode, where the SIP bridge and `rmsai-agent-phone` live. `webrtc`
    has no phone leg (the clinician joins from a browser), so it stays on the app's own LiveKit and
    agent, exactly as before. Dispatch is explicit: the worker registers under a name and won't
    auto-join, so it is requested into the per-event room before the call is placed or joined.
    """
    from voice.livekit_cloud import create_agent_dispatch  # noqa: PLC0415
    from voice.outbound import LiveKitCaller  # noqa: PLC0415

    if transport == "webrtc":
        # No phone: "answer" immediately so the alert is staged; the clinician joins over WebRTC.
        return (lambda room: SimulatedCaller([CallOutcome.ANSWERED]),
                lambda room: create_agent_dispatch(room, config=config))
    tel = config.telephony()
    return (lambda room: LiveKitCaller(tel, room=room),
            lambda room: create_agent_dispatch(room, config=tel))


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


def _report_call(result) -> None:
    """The dispatcher's line for one queued call (None = skipped, already logged)."""
    if result is None:
        return
    ob = result.outbound
    print(f"[call] {result.patient_ref}/{result.event_type} @ {result.bed} -> {ob.outcome} "
          f"(attempts {ob.attempts}) status={ob.status}", flush=True)


def _report(result, *, samples: bool = False) -> None:
    if result.trace is not None:
        from orchestrator.event_log import event_log_line  # noqa: PLC0415

        ob = result.outbound
        print(event_log_line(
            result.trace, app=result.app_dispatched if result.trace["gate"] else None,
            call=(ob.outcome if ob is not None and ob.called
                  else "queued" if result.call_queued else None),
            sms=ob.fallback if ob is not None else None, samples=samples), flush=True)
    tag = f"{result.patient_ref}/{result.event_type}"
    if result.call_queued:
        extra = " [app worklist pushed]" if result.app_dispatched else ""
        print(f"[queued] {tag} @ {result.bed}: call {result.queue_depth} in queue{extra}")
        return
    if not result.called:
        extra = " [app worklist pushed]" if result.app_dispatched else ""
        print(f"[skip] {tag} @ {result.bed}: {result.decision_reason}{extra}")
        return
    ob = result.outbound
    print(f"[{result.bed}] {tag} -> {ob.outcome} (attempts {ob.attempts}) status={ob.status}")
    for line in ob.transcript:
        print(f"    agent: {line}")


if __name__ == "__main__":
    raise SystemExit(main())
