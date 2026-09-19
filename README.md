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

`scripts/build.sh` compiles the two binaries for the cluster because their
kernels run there: the default is `-march=rv64imafdcv_zicsr_zifencei_zba_zbb_zbs_xsmtvdotii -mtune=spacemit-a100`, which names the A100 ISA and the vendor scheduling model
without the blanket `-mcpu` switch. `SPERT_A100_FLAGS` overrides it.

The matrix that chose it, five runs of `a100_bandwidth 64` each (four variants
per run, so a run counts only if all four completed):

| build | completed runs |
|---|---|
| plain `rv64gc` | 4 of 5 |
| `-mtune=spacemit-a100` only | 3 of 5 |
| `-march` without vector | 3 of 5 |
| `-march` with vector | 4 of 5 |
| `-march` with vector and `_xsmtvdotii` | 3 of 5 |
| `-mcpu=spacemit-a100` | **0 of 5** |

Two findings, and one correction to an earlier claim here. The correction: the
first measurement of this (three runs) reported "the flag decides whether they
run at all"; with five runs the truth is narrower - only the blanket
`-mcpu=spacemit-a100` fails deterministically, and every partial set behaves like
the plain build. That also means the tool is **intrinsically flaky**: even the
plain build fails roughly one run in five, which is an open defect of the tool or
its use of the runtime, not of any ISA option, and it is why the default now
avoids the switch that fails every time.

The roof is unaffected by all of it: plain and vector builds agree on a plain
stream at 24.6 GiB/s and on the decode path's own tiled walk near 23 GiB/s, so
the ceiling is the bus and not the instruction mix. The scalar nibble-unpack
variant reads 2.7 GiB/s in every build; answering whether the 4-bit format itself
can feed the bus still needs a hand-written vector unpack.

## One read serving R rows

`a100_bandwidth` also carries `gemvR1`, `gemvR4` and `gemvR8`: one pass over the
packed 4-bit weights, serving 1, 4 and 8 activation rows, each checked against a
host-side reference. Measured at a 1.5 GiB working set:

| variant | MiB/s |
|---|---|
| `gemvR1` | 2,610 |
| `gemvR4` | 2,481 |
| `gemvR8` | 2,547 |

The rate is flat as R grows, while the vendor's own kernels lose it: its one-row
`i8i4` kernel streams at 9.0 GiB/s and its four-row one at 5.8 GiB/s during a
decode run. That is the shape of the question this repository can answer for the
LLM work - the 25 % to 30 % batching loses to regrouping is a property of the
vendor kernel structure rather than of the memory path - with the caveat that
this kernel is scalar, so it shows R does not change *its* rate and not that it
reaches the vendor's. A vector-unpacking version is the next step, and the build
now accepts an A100 ISA that runs.

## Workspace entry

```xml
<project name="spine-runtime.git" path="projects/spine-runtime" />
```
