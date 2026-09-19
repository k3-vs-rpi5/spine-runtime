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

### An abnormal exit mid-launch takes the runtime down

Two attempts at that probe, both ending in the same place:

| event | symptom | recovery |
|---|---|---|
| the probe was `pkill`ed mid-run | every later stream refused (`reason=stream-refused`), `a100_probe` silent, TCM 8 of 8 free | reboot |
| the probe segfaulted | the same, including for the plain `a100_bandwidth 64` that had just worked | reboot |

After a reboot the cluster is healthy again (`a100_probe` reports
`kernels=ok mismatches=0 us_per_launch=31.2`; `a100_bandwidth 64` reports
`word 24,881 MiB/s`), so the state that breaks is runtime-side. The rules that
follow apply to every harness in this workspace: never kill a process with a
launch in flight, and after any abnormal exit re-establish cluster health before
trusting the next measurement rather than assuming either outcome - three
observations say a kill poisoned the runtime twice, a segfault once, and the
latest segfault not at all (`a100_probe` and `a100_bandwidth` were healthy right
after it).

The probe itself is unfinished, in three stages so far: it first spun (a one-tile
grid on an eight-core stream, no `future.valid()` check), then segfaulted with its
diagnostics lost to stdout buffering, then - with unbuffered stderr and a single
launch over all 32 cases - still segfaulted, with the output localising that to
the launch/sync boundary. Two concrete differences from every kernel here that
works point at the next experiments: this one executes `vmadot` (at VLEN=128,
SEW=8 the selected MAC unit is 2x2x4[x2], and the specification says an
unimplemented unit is an illegal instruction, so SEW=4 and SEW=16 variants and an
`e32, m1` accumulator would name the unit this cluster implements), and its
accumulator is a host-stack array where the working kernels write only into the
buffer the launch was given.

Both of those have now been answered by the probe itself. The host-stack
accumulator was the launch-time segfault: writing C into the buffer the launch was
given makes the launch complete. The specification's even-`vd` rule then bit once
more - **C occupies two sequential vector registers**, so a case needs 16 int32 at
VLEN=128, and the first working version left only one register's worth, which is
why its masks showed values above 1.

With 32 cases in one launch the mapping is measured, and it is not the naive one:
inputs of value 1 produce results of 1, 16, 17, 32, 33, 66 and 70, so **the MAC
consumes the operands in 4-bit lanes** even though the configuration asked for
`e8` (a byte 0x01 is read as nibbles (1,0), 0x10 as (0,1)). That is exactly why
the kernel written from the specification's prose produced four wrong answers out
of four. Interpreting the mapping now needs a designed run with distinctive
per-position values rather than bring-up work, and one caveat stays in the
record: the process still reports `free(): invalid next size` at exit after
printing the mapping, so the mapping is valid while the allocation discipline
around it is not yet settled.

Two changes then settled what could be settled. The probe's buffers are now
page-aligned mappings rather than vector storage, and the exit-time allocator
complaint disappeared - recorded as isolation, not as a diagnosis, since whether
the runtime writes past the buffer or crosses an allocator boundary is still
unknown. And the untouched operand now carries 1..16 by position, so a C value
names the position it paired with: in every A case one C slot carries a small
value that tracks the A position (2, 5, 7, 9, 11, ...), the large constants
(1496, 2992) are sums over many lanes, and the constant changes between A
positions 11 and 12 - the copy structure is real and asymmetric, as the
specification's sliding-packing note implies.

The full permutation is still not written down, and no rate is quoted until it
is: turning the pairing signal into a packing routine needs a deliberate solve
(single 1 in both operands over (p,q), reading back which C slot answers), not
another reading of the table.

That solve has now been run - 256 cases, a single 1 in A at p and in B at q, one
launch, clean exit - and its answer is not a permutation: only 33 of the 256 cases
produce output, up to 12 C slots light per case, and **only the first two bytes of
A participate at all** (p = 0 and 1 multiply with every q; p >= 2 produces
nothing). One input 1 lighting many accumulators matches the specification's
"sliding-packing" wording and the earlier 4-bit-lane finding, and the two-byte
operand consumption, if it holds, would make a hand-written vmadot kernel on the
A100 need many instructions per weight byte - the opposite of the vendor's 9.0
and 5.8 GiB/s there.

That is not yet a conclusion: the sweep used one configuration (`e8` with an
`e32, m2` accumulator). The `e4`, `e16` and different-`vl` variants are what would
say whether the small operand consumption is a property of this configuration or
of the cluster. Recorded as the state of the A100 own-kernel route: a measured
obstacle, a few configuration runs away from being settled.

### The obstacle is withdrawn: every byte of A participates

The configuration runs have been made, and they refute the paragraph above. With
a single 1 in A at p and **B all ones** - a different question from the pairing
sweep - both full-vector configurations show **all sixteen A positions live**
(`e8`: 0-15; `e16`: 0-15), and the AVL=8 row shows the window shrinking as a
smaller `vl` should (0-7 minus 6). So the earlier sparsity was a property of the
pairing, not of operand consumption: with a single 1 in each operand, only the A
positions whose partner lands inside the 16-byte B window can answer.

Two toolchain facts are recorded with it. The specification's SEW=4 configuration
cannot be requested through this assembler at all (`vsetvli ..., e4, m1` is
"illegal operands", and the immediate-AVL form is `vsetivli`, not `vsetvli`), so
the smallest MAC unit would need raw instruction encoding. And a failed compile
used to leave the previous binary in the stage - which is how a sweep once ran
with no output and looked like a silent result; `build.sh` now removes both
binaries before compiling.

What remains for the A100 own-kernel route is a pairing solve with a B window wide
enough to hold every partner of every A position, then the packing routine, then
the rate at R=1, 2 and 4 against the vendor's 9.0 and 5.8 GiB/s.

### That solve fixes the operand size and shows the layout is a broadcast

Run as described - 64-byte cases with A across two registers and B in a third, a
single 1 in each operand, 512 combinations - the answers refute the assumption the
widened window was built on and settle the operand sizes:

- **A is one register, not two.** A positions 16..31 never answer, so at VLEN=128
  and SEW=8 an instruction consumes 16 bytes of A and 16 of B - exactly the
  specification's 2x2x4 with Copies=2 at this vector length.
- **The operand is broadcast, not mapped one-to-one.** Every A position pairs with
  nearly every B position, and the lit C slots form a small set (`{0,2,6}`, `{2,6}`)
  rather than one slot per pair: a byte is slid across several MAC lanes, which is
  what "sliding-packing" means and why a packing written from the prose is wrong.

Two routes to the packing remain, both concrete: fit the layout from a known GEMM
by comparing an all-ones run against candidate layouts, or - shorter - disassemble
the vendor's own i8i4 kernels in the board's `libllama-bench-impl.so`, which are
the code that actually feeds this silicon at 9.0 GiB/s. No rate is claimed until one
of them lands.

## Workspace entry

```xml
<project name="spine-runtime.git" path="projects/spine-runtime" />
```
