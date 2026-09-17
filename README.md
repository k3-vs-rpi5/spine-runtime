[中文](README.zh-CN.md)

# Spine-Runtime project

The vendor's own programming model for the AI cluster, and the only documented
way to submit work to it. `libspert` is prebuilt; the API and the scheduling
model are public.

## Pinned source

| field | value |
|---|---|
| upstream | `https://github.com/spacemit-com/spine-runtime.git` |
| ref | `main` |
| commit | `b3ff5182de116613b6db72b0f43064554ac3d42d` |
| license | MIT |
| checkout | `src/spine-runtime` |
| SDK package | `spine-runtime.riscv64.0.6.2.tar.gz`, `libspert.so.0.6.2` |

The model is Tile-SPMD: a `Stream` holds compute cores granted by a backend, a
`Grid` splits the work into tiles, and each tile reads its coordinate from a
`Context`. `backend_info()` reports the granted core count, shared-buffer size,
vector length and architecture id. The `spacemit-k3` backend targets the AI
cores 8-15 with a 384 KiB per-core buffer; `spacemit-k3-x100` is a compatible
mode on the X100 cores and `generic/qemu` is for functional validation only.

## Status

- documentation repository and SDK identity synced; the SDK stays a sealed
  input outside Git, with its digest in `sources.lock.yaml`.
- `spacemit-runtime` is not installed on the board yet, so a first A100 program
  still needs that package or the prebuilt SDK.
- automation exists: `scripts/build.sh` cross-builds `src/spine_runtime_project/a100_probe.cc` against
  the SDK and the board's `spine_tcm` header, and `scripts/run.sh` delivers and
  runs it, keeping the output as evidence.
- measured on the board: the default backend reports `arch_id 0xa064` (A100),
  eight granted cores, and 59.9 % AI-cluster busy by `/proc/stat` during a
  900000-launch window, agreeing with the per-CPU counters.

## Workspace entry

```xml
<project name="spine-runtime.git" path="projects/spine-runtime" />
```
