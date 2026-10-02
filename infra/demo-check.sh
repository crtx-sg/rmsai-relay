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
fail=0

ok()  { printf '  %-13s ok    %s\n' "$1" "${2:-}"; }
bad() { printf '  %-13s FAIL  %s\n' "$1" "$2"; fail=1; }

# Container id for a compose service, or empty.
cid() { docker compose -f infra/docker-compose.yml ps -q "$1" 2>/dev/null; }
state() {
  docker inspect -f '{{.State.Status}}{{if .State.Health}}/{{.State.Health.Status}}{{end}}' "$1" 2>/dev/null
}

echo "containers:"
for svc in redis neo4j qdrant livekit consumer voice-worker gateway; do
  id=$(cid "$svc")
  if [ -z "$id" ]; then bad "$svc" "not created -> make demo-up"; continue; fi
  st=$(state "$id")
  case "$st" in
    running|running/healthy) ok "$svc" "$st" ;;
    *) bad "$svc" "$st -> docker compose -f infra/docker-compose.yml logs $svc" ;;
  esac
done

echo "endpoints:"
python3 -c "import socket;socket.create_connection(('localhost',6379),2).close()" 2>/dev/null \
  && ok redis :6379 || bad redis ":6379 unreachable -> make stores-up"
curl -fsS -m 2 -o /dev/null http://localhost:7474 && ok neo4j :7474 || bad neo4j ":7474 unreachable -> make stores-up"
curl -fsS -m 2 -o /dev/null http://localhost:6333/collections && ok qdrant :6333 || bad qdrant ":6333 unreachable -> make stores-up"
curl -fsS -m 2 -o /dev/null http://localhost:7880 && ok livekit :7880 || bad livekit ":7880 unreachable -> make stores-up"
body=$(curl -sS -m 2 "http://localhost:$WORKER_PORT/" 2>&1)
[ "$body" = "OK" ] && ok voice-worker ":$WORKER_PORT health OK" \
  || bad voice-worker ":$WORKER_PORT says '$body' -> docker restart $(cid voice-worker | cut -c1-12)"
curl -fsS -m 2 -o /dev/null "http://localhost:$GATEWAY_PORT/" && ok gateway ":$GATEWAY_PORT" \
  || bad gateway ":$GATEWAY_PORT unreachable -> docker compose -f infra/docker-compose.yml logs gateway"

echo "livekit registration:"
# The worker must have registered with the CURRENT LiveKit process. Started-before-LiveKit is fine
# if it reconnected (it does, within its 16 retries); it is not if it gave up first.
lk=$(cid livekit); vw=$(cid voice-worker)
if [ -n "$lk" ] && [ -n "$vw" ]; then
  lk_t=$(docker inspect -f '{{.State.StartedAt}}' "$lk")
  if docker logs --since "$lk_t" "$vw" 2>&1 | grep -q '"registered worker"'; then
    ok voice-worker "registered since livekit started (${lk_t%.*})"
  else
    bad voice-worker "no registration since livekit started ${lk_t%.*} -> docker restart ${vw:0:12}"
  fi
fi

echo
if [ $fail -eq 0 ]; then
  echo "READY. Open http://localhost:$GATEWAY_PORT (the agent joins the inbox room when the app logs in)."
else
  echo "NOT READY - fix the FAIL lines above (or re-run: make demo-up)."
fi
exit $fail
