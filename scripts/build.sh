#!/bin/sh
# Cross-build the A100 bring-up probe against the published Spine-Runtime SDK.
#
# The probe source lives in src/spine_runtime_project/ so this project keeps the
# same shape as the other projects: pinned upstream in src/<upstream>, our
# automation in src/<project>_project, and only two public scripts.
#
# Binary-only delivery: the board receives the probe and one shared library; it
# needs no compiler, no SDK and no package manager.
set -eu

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname "$0")" && pwd)
ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE=$(CDPATH='' cd -- "$ROOT/../.." && pwd)
TOOLCHAIN=${SPERT_TOOLCHAIN:-$WORKSPACE/spacemit-toolchain-linux-glibc-x86_64-v1.2.4}
SDK=${SPERT_SDK:-$WORKSPACE/.local/k3-tools/riscv64/spert}
TCM=${SPERT_TCM:-$WORKSPACE/.local/k3-tools/riscv64/spine-tcm}
CXX="$TOOLCHAIN/bin/riscv64-unknown-linux-gnu-g++"
OUT=${SPERT_OUT:-$WORKSPACE/.local/spine-runtime/build}
STAGE=${SPERT_STAGE:-$WORKSPACE/.local/spine-runtime/stage}

[ -x "$CXX" ] || { echo "missing workspace cross toolchain: $CXX" >&2; exit 1; }
[ -f "$SDK/include/spert.hpp" ] || { echo "missing Spine-Runtime SDK: $SDK" >&2; exit 1; }
[ -f "$TCM/include/spine_tcm.h" ] || {
    echo "missing the board's spine_tcm header copy: $TCM" >&2; exit 1; }
mkdir -p "$OUT" "$STAGE"

echo "== build the A100 probe (riscv64) =="
"$CXX" --sysroot="$TOOLCHAIN/sysroot" -O2 -std=c++17 -Wall \
    -I"$ROOT/src" -I"$SDK/include" -I"$TCM/include" \
    -o "$OUT/a100_probe" "$ROOT/src/spine_runtime_project/a100_probe.cc" \
    -L"$SDK/lib" -lspert -Wl,-rpath,'$ORIGIN' -pthread
"$CXX" --sysroot="$TOOLCHAIN/sysroot" -O2 -std=c++17 -Wall \
    -I"$ROOT/src" -I"$SDK/include" -I"$TCM/include" \
    -o "$OUT/a100_bandwidth" "$ROOT/src/spine_runtime_project/a100_bandwidth.cc" \
    -L"$SDK/lib" -lspert -Wl,-rpath,'$ORIGIN' -pthread
"$CXX" -print-file-name=libstdc++.so.6 >/dev/null
cp "$OUT/a100_probe" "$STAGE/a100_probe"
cp "$SDK/lib/libspert.so.1" "$STAGE/libspert.so.1"
chmod +x "$STAGE/a100_probe"
printf "  %-14s %s bytes\n" a100_probe "$(stat -c%s "$STAGE/a100_probe")"
cp "$OUT/a100_bandwidth" "$STAGE/a100_bandwidth"
chmod +x "$STAGE/a100_bandwidth"
printf "  %-14s %s bytes\n" a100_bandwidth "$(stat -c%s "$STAGE/a100_bandwidth")"
printf "  %-14s %s bytes (SDK %s)\n" libspert.so.1 "$(stat -c%s "$STAGE/libspert.so.1")" \
    "$(cat "$SDK/VERSION_NUMBER" 2>/dev/null || echo unknown)"
if command -v file >/dev/null 2>&1; then
    file "$STAGE/a100_probe" | cut -c1-90
fi
echo "== payload =="
( cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
sed 's/^/  /' "$STAGE/SHA256SUMS"
