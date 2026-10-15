#!/usr/bin/env bash
# Is the relay running the vendored ecgtranscnn at its pin, with the intended weights? Exit 0 = yes.
#
# The model code lives in two places that drift independently: the host .venv (editable link to
# external/ecgtranscnn, live) and the Docker image (a copy baked at `make docker-build`). Weights
# are read live from ECG_CHECKPOINTS in .env on both. Each FAIL line names the fixing command.
set -u
cd "$(dirname "$0")/.."

VENDOR=external/ecgtranscnn
PY=.venv/bin/python
ECGTRANSCNN_DIR=${ECGTRANSCNN_DIR:-../ecgtranscnn}
COMPOSE="docker compose -f infra/docker-compose.yml"
fail=0

ok()   { printf '  %-18s ok    %s\n' "$1" "${2:-}"; }
bad()  { printf '  %-18s FAIL  %s\n' "$1" "$2"; fail=1; }
skip() { printf '  %-18s skip  %s\n' "$1" "$2"; }

# sha256 over every .py file of the importable ecg_transcovnet package (path + content).
SRC_HASH='import hashlib, pathlib, ecg_transcovnet as m
root = pathlib.Path(m.__file__).resolve().parent
h = hashlib.sha256()
for p in sorted(root.rglob("*.py")):
    h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
print(h.hexdigest()[:16])'

echo "ecgtranscnn code:"
pin=$(sed -n 's/.*\*\*Pinned commit:\*\* `\([0-9a-f]*\)`.*/\1/p' "$VENDOR/PLACEHOLDER.md")
head=$(git -C "$VENDOR" rev-parse HEAD 2>/dev/null)
if [ -z "$pin" ]; then
  bad pin "no **Pinned commit:** in $VENDOR/PLACEHOLDER.md"
elif [ -z "$head" ]; then
  bad pin "$VENDOR is not a git clone -> re-clone it (see PLACEHOLDER.md)"
elif [ "${head#"$pin"}" = "$head" ]; then
  bad pin "$VENDOR is at ${head:0:7}, pin is $pin -> git -C $VENDOR checkout $pin && make external"
elif [ -n "$(git -C "$VENDOR" status --porcelain --untracked-files=no)" ]; then
  bad pin "$VENDOR has local edits to tracked files -> git -C $VENDOR diff"
else
  ok pin "$pin"
fi

host_file=$($PY -c 'import os, ecg_transcovnet as m; print(os.path.realpath(m.__file__))' 2>/dev/null)
case "$host_file" in
  "$(realpath "$VENDOR")"/*) ok host-venv "imports $VENDOR" ;;
  "") bad host-venv "ecg_transcovnet not importable in $PY -> make external" ;;
  *) bad host-venv "imports $host_file -> make external" ;;
esac
host_hash=$($PY -c "$SRC_HASH" 2>/dev/null)

if ! command -v docker >/dev/null 2>&1; then
  skip docker-image "docker not installed"
else
  image_hash=$($COMPOSE run --rm --no-deps -T tools python -c "$SRC_HASH" 2>/dev/null | tail -1)
  if [ -z "$image_hash" ]; then
    bad docker-image "could not run the tools image -> make docker-build"
  elif [ "$image_hash" != "$host_hash" ]; then
    bad docker-image "baked code differs from $VENDOR -> make docker-build"
  else
    ok docker-image "same code as $VENDOR"
  fi
  # Running app containers (any profile) keep the image they were created from until recreated.
  app_image=rmsai-relay/app:local
  want=$(docker image inspect -f '{{.Id}}' "$app_image" 2>/dev/null)
  for id in $(docker ps -q --filter "label=com.docker.compose.project=infra"); do
    [ "$(docker inspect -f '{{.Config.Image}}' "$id")" = "$app_image" ] || continue
    svc=$(docker inspect -f '{{index .Config.Labels "com.docker.compose.service"}}' "$id")
    if [ "$(docker inspect -f '{{.Image}}' "$id")" = "$want" ]; then
      ok "$svc" "running the current image"
    else
      bad "$svc" "running an older image -> make demo-up (or demo-phone-up) to recreate it"
    fi
  done
fi

echo "checkpoints (ECG_CHECKPOINTS in .env):"
ckpts=$($PY -c 'from common.config import DEFAULT as c; print("\n".join(c.ecg_checkpoints))' 2>/dev/null)
if [ -z "$ckpts" ]; then
  bad checkpoints "ECG_CHECKPOINTS unset -> the deterministic STUB runs, predictions are fake"
else
  while IFS= read -r ck; do
    name=${ck#"$VENDOR"/}
    if [ ! -f "$ck" ]; then
      bad "$(basename "$ck")" "missing $ck -> make weights"
    elif [ -f "$ECGTRANSCNN_DIR/$name" ]; then
      if [ "$(sha256sum < "$ck")" = "$(sha256sum < "$ECGTRANSCNN_DIR/$name")" ]; then
        ok "$(basename "$ck")" "same as $ECGTRANSCNN_DIR/$name"
      else
        bad "$(basename "$ck")" "differs from $ECGTRANSCNN_DIR/$name -> make weights"
      fi
    else
      ok "$(basename "$ck")" "present (no $ECGTRANSCNN_DIR copy to compare)"
    fi
  done <<< "$ckpts"
fi

exit $fail
