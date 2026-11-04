#!/usr/bin/env bash
# Is the Docker demo (DEMO.md) up, healthy and started in the right order? Exit 0 = ready.
#
# Each line names the failing piece and the command that fixes it. Checks run in dependency order:
# backing stores -> LiveKit -> voice worker (registered with LiveKit, started AFTER it) -> consumer
# -> gateway. A worker that started before LiveKit answers retries 16x, gives up, and stays "Up"
# with no connection: events still reach the app, but nothing speaks and no chat gets a reply.
set -u

GATEWAY_PORT=${GATEWAY_PORT:-8080}
WORKER_PORT=${LIVEKIT_WORKER_HTTP_PORT:-8091}
# A second hospital (COMPOSE_PROJECT_NAME=infra-h2, infra/h2.env) has its own stores on its own
# ports but no LiveKit container: it uses h1's (project `infra`).
PROJECT=${COMPOSE_PROJECT_NAME:-infra}
SERVICES="redis neo4j qdrant livekit consumer voice-worker gateway"
[ "$PROJECT" = infra ] || SERVICES="redis neo4j qdrant consumer voice-worker gateway"
fail=0

ok()  { printf '  %-13s ok    %s\n' "$1" "${2:-}"; }
bad() { printf '  %-13s FAIL  %s\n' "$1" "$2"; fail=1; }

# Container id for a compose service, or empty.
cid() { docker compose -f infra/docker-compose.yml -p "$PROJECT" ps -q "$1" 2>/dev/null; }
state() {
  docker inspect -f '{{.State.Status}}{{if .State.Health}}/{{.State.Health.Status}}{{end}}' "$1" 2>/dev/null
}

echo "containers ($PROJECT${HOSPITAL_ID:+, hospital $HOSPITAL_ID}):"
for svc in $SERVICES; do
  id=$(cid "$svc")
  if [ -z "$id" ]; then bad "$svc" "not created -> make demo-up"; continue; fi
  st=$(state "$id")
  case "$st" in
    running|running/healthy) ok "$svc" "$st" ;;
    *) bad "$svc" "$st -> docker compose -f infra/docker-compose.yml logs $svc" ;;
  esac
done

echo "endpoints:"
RP=${REDIS_PORT:-6379}; NP=${NEO4J_HTTP_PORT:-7474}; QP=${QDRANT_HTTP_PORT:-6333}
python3 -c "import socket;socket.create_connection(('localhost',$RP),2).close()" 2>/dev/null \
  && ok redis ":$RP" || bad redis ":$RP unreachable -> make stores-up"
curl -fsS -m 2 -o /dev/null "http://localhost:$NP" && ok neo4j ":$NP" || bad neo4j ":$NP unreachable -> make stores-up"
curl -fsS -m 2 -o /dev/null "http://localhost:$QP/collections" && ok qdrant ":$QP" || bad qdrant ":$QP unreachable -> make stores-up"
curl -fsS -m 2 -o /dev/null http://localhost:7880 && ok livekit :7880 || bad livekit ":7880 unreachable -> make stores-up"
body=$(curl -sS -m 2 "http://localhost:$WORKER_PORT/" 2>&1)
[ "$body" = "OK" ] && ok voice-worker ":$WORKER_PORT health OK" \
  || bad voice-worker ":$WORKER_PORT says '$body' -> docker restart $(cid voice-worker | cut -c1-12)"
curl -fsS -m 2 -o /dev/null "http://localhost:$GATEWAY_PORT/" && ok gateway ":$GATEWAY_PORT" \
  || bad gateway ":$GATEWAY_PORT unreachable -> docker compose -f infra/docker-compose.yml logs gateway"

echo "livekit registration:"
# The worker must have registered with the CURRENT LiveKit process. Started-before-LiveKit is fine
# if it reconnected (it does, within its 16 retries); it is not if it gave up first.
lk=$(PROJECT=infra cid livekit); vw=$(cid voice-worker)
if [ -n "$lk" ] && [ -n "$vw" ]; then
  lk_t=$(docker inspect -f '{{.State.StartedAt}}' "$lk")
  # Right after `up --wait` the worker can be healthy a few seconds before it logs registration.
  for _ in $(seq 20); do
    docker logs --since "$lk_t" "$vw" 2>&1 | grep -q '"registered worker"' && break; sleep 1
  done
  if docker logs --since "$lk_t" "$vw" 2>&1 | grep -q '"registered worker"'; then
    ok voice-worker "registered since livekit started (${lk_t%.*})"
  else
    bad voice-worker "no registration since livekit started ${lk_t%.*} -> docker restart ${vw:0:12}"
  fi
fi

if [ "$PROJECT" != infra ]; then
  echo "agent names (must differ from h1, or h1's calls can reach this stack's worker):"
  envv() { grep -E "^$1=" .env 2>/dev/null | tail -1 | cut -d= -f2 | awk '{print $1}'; }
  h1a=$(envv LIVEKIT_AGENT_NAME); h1a=${h1a:-rmsai-agent}
  h1s=$(envv LIVEKIT_SIP_AGENT_NAME); h1s=${h1s:-rmsai-agent-phone}
  [ -n "${LIVEKIT_AGENT_NAME:-}" ] && [ "$LIVEKIT_AGENT_NAME" != "$h1a" ] \
    && ok agent "$LIVEKIT_AGENT_NAME" || bad agent "'${LIVEKIT_AGENT_NAME:-}' same as h1 -> fix infra/h2.env"
  [ -n "${LIVEKIT_SIP_AGENT_NAME:-}" ] && [ "$LIVEKIT_SIP_AGENT_NAME" != "$h1s" ] \
    && ok phone-agent "$LIVEKIT_SIP_AGENT_NAME" || bad phone-agent "'${LIVEKIT_SIP_AGENT_NAME:-}' same as h1 -> fix infra/h2.env"
fi

echo
if [ $fail -eq 0 ]; then
  echo "READY. Open http://localhost:$GATEWAY_PORT (the agent joins the inbox room when the app logs in)."
else
  echo "NOT READY - fix the FAIL lines above (or re-run: make demo-up)."
fi
exit $fail
