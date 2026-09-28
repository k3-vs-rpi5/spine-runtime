#!/bin/sh
# Deliver the A100 probe to the board, run it there, and keep its output as
# evidence. Board credentials come from the workspace record .ssh-k3 and are
# never printed.
#
#   scripts/run.sh [--evidence DIR]
set -eu

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname "$0")" && pwd)
ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE=$(CDPATH='' cd -- "$ROOT/../.." && pwd)
STAGE=${SPERT_STAGE:-$WORKSPACE/.local/spine-runtime/stage}
CREDENTIALS=${SPERT_CREDENTIALS:-"$WORKSPACE/.ssh-k3"}
EVIDENCE=

while [ $# -gt 0 ]; do
    case "$1" in
        --evidence) EVIDENCE=$2; shift 2 ;;
        *) echo "usage: $0 [--evidence DIR]" >&2; exit 2 ;;
    esac
done

[ -d "$STAGE" ] || { echo "stage missing: run scripts/build.sh first" >&2; exit 1; }
command -v sshpass >/dev/null 2>&1 || { echo "sshpass is required" >&2; exit 1; }
[ -r "$CREDENTIALS" ] || { echo "cannot read credentials: $CREDENTIALS" >&2; exit 1; }
mode=$(stat -c '%a' "$CREDENTIALS")
[ "$mode" = "600" ] || { echo "credentials must be mode 0600 (found $mode)" >&2; exit 1; }

host=; user=; secret=
while read -r host_field user_field secret_field _rest; do
    host=$host_field; user=$user_field; secret=$secret_field; break
done < "$CREDENTIALS"
if [ -z "$host" ] || [ -z "$user" ] || [ -z "$secret" ]; then
    echo "credentials record must contain: host account secret" >&2
    exit 1
fi

PAYLOAD_ID=$(cd "$STAGE" && find . -type f -print0 | sort -z | xargs -0 cat | sha256sum | cut -c1-8)
REMOTE=/tmp/spine-a100-$PAYLOAD_ID

SSHPASS=$secret
export SSHPASS
# A board whose address changed is reached without editing the credential
# record: export K3_BOARD_HOST=<address> for this invocation.
[ -n "${K3_BOARD_HOST:-}" ] && host=$K3_BOARD_HOST
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$WORKSPACE/.local/known_hosts \
-o ConnectTimeout=20 -o NumberOfPasswordPrompts=1 -o PreferredAuthentications=password,publickey"
DEST="$user@$host"

remote() { sshpass -e ssh $SSH_OPTS "$DEST" "$@"; }

remote "rm -rf $REMOTE && mkdir -p $REMOTE"
sshpass -e scp -q -r $SSH_OPTS "$STAGE/." "$DEST:$REMOTE/"
remote "chmod +x $REMOTE/a100_probe"

LOG=$(mktemp)
{
    echo "# board: $(remote 'uname -sr; cat /proc/device-tree/model 2>/dev/null | tr -d "\0"; echo')"
    echo "# payload: $PAYLOAD_ID"
    echo "# spine_tcm-smi:"
    remote "spacemit-tcm-smi -i 2>&1 | head -3"
    echo "# probe output:"
    remote "cd $REMOTE && LD_LIBRARY_PATH=$REMOTE ./a100_probe 2>&1"
    echo "# bandwidth output:"
    remote "cd $REMOTE && LD_LIBRARY_PATH=$REMOTE ./a100_bandwidth 2>&1"
} > "$LOG" 2>&1 || true
cat "$LOG"

if [ -n "$EVIDENCE" ]; then
    mkdir -p "$EVIDENCE"
    cp "$LOG" "$EVIDENCE/a100_probe.log"
    (cd "$EVIDENCE" && find . -type f ! -name SHA256SUMS -print0 | sort -z |
        xargs -0 sha256sum > SHA256SUMS)
    echo "evidence written to $EVIDENCE"
fi
