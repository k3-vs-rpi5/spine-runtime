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
- measured on the board: the default backend reports `arch_id 0xa064` (A100) and
  grants eight cores (ids 8-15), and a 900000-launch window is evidenced from the
  vendor side - the runtime's registration of that work and the TCM blocks
  granted to the submitting pid - together with the throughput arithmetic the
  kernel's own size and launch count imply. Per-CPU counters and `/proc/stat` are
  recorded but are not A100 evidence here: measured on 2026-09-18, the cluster's
  work does not move them.

## The A100 ISA flag and the kernels we submit

`scripts/build.sh` compiles the two binaries for the cluster
(`-mcpu=spacemit-a100`, overridable with `SPERT_A100_FLAGS`) because their kernels
run there. Measured on 2026-09-19, that flag decides whether they run at all:

| build | `a100_bandwidth 64` completed runs |
|---|---|
| plain `rv64gc` | 3 of 3 |
| `-mcpu=spacemit-a100` | 0 of 3 |
| `-mcpu=spacemit-a100 -fno-tree-vectorize` | 0 of 3 |

The vector-free build crashes too, so this is not the auto-vectoriser; the simple
probe kernel survives either way, so it is not the flag breaking every kernel.
Until that is understood, the bandwidth numbers below are scalar-code numbers -
and they are still worth having, because a scalar streaming kernel already
reaches **24,627 MiB/s** at a 1.5 GiB working set, which is the same ceiling the
X100-side probe reports. The memory roof therefore stands whatever the
instruction mix is, and the decode path's own 65,792-byte tiled walk - 22,658 MiB/s,
8 % below a plain stream - is what serial decode is measured against.

## Workspace entry

```xml
<project name="spine-runtime.git" path="projects/spine-runtime" />
```
