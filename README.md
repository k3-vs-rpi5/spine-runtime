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

## Do not ask the compiler to vectorise these kernels

Trying to shortcut the vector-unpacking work with `-O3 -mrvv-vector-bits=zvl`
produces **wrong answers for every variant**, including the trivial `word` and
`byte` loops - caught only because each variant is checked against a host-side
reference. `zvl` takes the vector length from the `-march` string, and the vendor
extension set carries `zvl1024b` while the AI cores have `vlen=128`, so the
compiler targets a machine that is not there; GCC 15 accepts only `scalable` or
`zvl`, so 128 cannot be named directly either. The build is not faster:
`word` falls from 24,759 to 9,741 MiB/s. A hand-written kernel is the only way
left to that question, and it has to carry its own checksum.

## The hand-written kernel does not reach the vendor's rate either

That hand-written kernel now exists for the unpack part (`pack4v`, RVV
intrinsics, host-verified) and it answers the question the other way:

| variant | MiB/s |
|---|---|
| `word` (plain stream) | 24,168 |
| `tiled` | 23,233 |
| `pack4` (scalar unpack) | 2,728 |
| `pack4v` (RVV unpack) | 2,536 |

The vector unpack is **slower** than the scalar loop, and both are 3.3 to 3.6 times
below the vendor's one-row kernel (9.0 GiB/s). The vendor's advantage is almost
certainly the IME2 instruction family, which the toolchain can emit but whose
programming model is not documented in this workspace. So the structural finding
(that batching loses 25 % to 30 % to the vendor's kernel structure rather than to
the memory path) stands, but cashing it would need IME2 documentation rather
than more measurement here. Two side results are kept in the record: a lane-wise
reduction version was faster and wrong (the mismatch equalled one pass of nibble
sums), and the tool now prints expected and got on a mismatch instead of only
"mismatch".

## The IME documentation exists, and the instructions run on the X100 cores

The paragraph above was written before looking for the specification. It does
exist - `spacemit-com/riscv-ime-extension-spec` publishes the instructions
(`vmadot`, `vmadot1`/`2`/`3`/`n`, `vmadotu`/`su`/`us`, and the float
`vfmadot*` family), their types (int4/int8/int16, fp4/fp8/fp16/bf16), the MAC
unit selected from `vl*SEW`, and the programmer's model (RVV registers, no added
state, VLEN 128 to 4096, SEW 4/8/16, LMUL <= 1) - and its
`example/vmadot-gemm-demo.c` is a self-testing 4x8x4 int8 GEMM.

Measured here: the example compiles with `-march=rv64gcv_xsmtvdotii` (plain
`-march=rv64gcv` refuses the mnemonic) and prints
`Test successful. CRef equal to C.` on the **X100 cores**, under `taskset -c 0`
as well. A loop of the same shape
measures 786 Mmac/s scalar against 18,286 Mmac/s with `vmadot` - 23 times - with
**different results**, because the MAC unit at VLEN=128 and SEW=8 is 2x2x4[x2]
and the operands have to be packed the way the example's `Gemm_packB` does.

So the claim that IME2 is undocumented and reachable only through the AI-cluster
runtime does not hold for this board: the instruction set is published and
executes as ordinary userspace code on the Linux cluster. Writing our own
multi-row kernel is open again, with a packing routine to write and a self-test to
pass first.

## What such a kernel measures

The demo's shape, with its B operand packed once and streamed from a large buffer
while the accumulator stays in a vector register, on **one** X100 core pinned with
`taskset -c 0`:

| weight buffer | checksum | rate |
|---|---|---|
| 256 MiB | ok | 8,029 MiB/s |
| 1 GiB | ok | 7,832 MiB/s |

Against the other streamers: our scalar unpack reads 2.7 GiB/s and our RVV unpack
2.5 GiB/s on all eight AI cores, the vendor's one-row kernel 9.0 GiB/s per kernel
on the AI cluster. So one X100 core is within 15 % of the vendor's per-kernel
rate, eight of them would ask for about 61 GiB/s against a 24.6 GiB/s roof, the
MAC unit already consumes a four-row activation tile per weight tile (the
structure the vendor loses 25 % to 30 % on when it regroups a batch), and all of
it is ordinary threads on cpu0-7 where `taskset`, per-process accounting, perf and
the eBPF collectors apply.

Also recorded: `csrr vlenb` reads 32 bytes on the X100 cores, so **VLEN=256**,
while the AI cluster's backend reports 128 - the two clusters are not
interchangeable, which is why the demo (written for VLEN=256) passes here.

Not yet done: a multi-core version, a decode-shaped K, and any end-to-end token
rate.

## The A100 needs its own layout work, and that is measured rather than read

The decode work runs on the A100 cluster, so the kernel that matters is an A100
one - and the A100 has **VLEN=128** against the X100's 256. The IM
specification's `Copies` rule makes that difference structural:
`Copies = (sqrt(VLEN/64) is an integer) ? 1 : 2`, so VLEN=128 gives Copies=2 and
VLEN=256 gives Copies=1, and with them different operand packings (at VLEN=128,
SEW=8 selects a 2x2x4 MAC unit; at VLEN=256 it is 4x4x8).

Two consequences are recorded rather than assumed: the vendor demo, verified
correct here, is a **VLEN=256** kernel and its layout does not transfer; and a
kernel written from the specification's prose alone - A as `Copies*(M,K)`
row-major, B as `Copies*(K,N)` K-major, one `vmadot` per 2x2 output with K=8 -
compiles and runs on the A100 but does **not** reproduce the reference (four of
four outputs differ). The specification's operand figures are images and its
integer pseudocode indexes C as an (M,N) matrix with `i*K + j`, which cannot hold
for M != N.

The way through is to measure the permutation with a layout probe submitted
through `spert` on cores 8-15, exactly as this workspace measured how tracepoint
programs attach: source as hypothesis, probe as fact. Then the packing, then the
rate at R=1, 2 and 4 against the vendor's 9.0 and 5.8 GiB/s.

## Workspace entry

```xml
<project name="spine-runtime.git" path="projects/spine-runtime" />
```
