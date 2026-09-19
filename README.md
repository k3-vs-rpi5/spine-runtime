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

## Public material

The runtime does not stop at its headers. Published, and checked here:

| what | where |
|---|---|
| documentation set, English and Chinese | `README.md`, `README_ZH.md`, `docs/api-reference(_ZH).md`, `docs/programming-model*`, `docs/scheduling*`, `docs/backends*` |
| worked example | `examples/quickstart/` (`demo.cpp`, `CMakeLists.txt`, README in both languages) |
| releases carrying SDK assets | `0.6.0` (3 assets), `0.6.1` (2), `0.6.2` (2 - the SDK this project pins) |
| the AI cluster's instruction set | `spacemit-com/docs-ai`, `en|zh/architecture/ime_extension.md`: `Zvvm_spacemit` profile v0.6, "Status: Public Release", 2026-04-13; cloned in this workspace at `.local/docs-ai` (commit `2b3894e`) |

The instruction-set document is the one that decides kernel work: it defines the
matrix family the vendor's own decode kernel runs, block-scaled path included, with
semantics and formal encodings. See the correction at the end of this file.

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

### The second route answers it, and changes the question

The kernels are not in `libllama-bench-impl.so` but in the board's
`libggml-cpu.so.0.16.0`, which is not stripped and exports
`spacemit_kernels::ime2::gemm_kernel_i8i4` plus its `_m1` and `_m4` leaves (the
dispatcher is a stub that picks between them on a row-count threshold, exactly the
rule the kernel probe saw from outside). Disassembling the decode leaf:

```
c6850:  vsetvli t0,zero,e8,m1,tu,mu
c6854:  vl4r.v  v4,(s5)          # segment load of four registers
c685c:  vsetvli t0,zero,e8,mf2,tu,mu
c6860:  vle8.v  v30,(s4)
c687c:  flw     ft0,0(s2)        # the scale
c683a:  vfcvt.f.x.v v0,v0        # dequantise to float
c6908:  vfmacc.vf v2,ft0,v31     # floating-point multiply-accumulate
```

**`vmadot` appears zero times in the whole library**, and the rare-mnemonic
histogram holds nothing outside the standard RVV and scalar sets. The name `ime2`
describes the build configuration and the core it targets, not a matrix
instruction: the vendor reaches 9.0 GiB/s on the AI cluster with hand-written RVV
that loads 4-bit weights in 64-byte segments, converts them in registers and
accumulates with `vfmacc`.

That removes the obstacle the last three rounds were chasing, and it says what to
do instead: write an RVV gemv in the vendor's shape - segmented 4-bit loads,
in-register dequantisation, `vfmacc` accumulation, four rows per weight read -
verify it against a scalar reference, and measure it against that 9.0 GiB/s. The
IME layout work is not on that path any more.

### The RVV gemv is written, verified, and 4.5 times behind

`a100_bandwidth` now carries `rvvgemv`: standard RVV in the vendor's shape, 4-bit
blocks streamed 16 bytes at a time, each block multiplied by its scale and
accumulated, checked against a host reference. It needed one fix worth recording -
the u32 *reduction* sums 16 lanes, so the drain window has to fit 16 lanes' worth
in u32 rather than one lane's; draining every 1,024 blocks overflowed (the
reference caught it), every 64 does not.

At a 1536 MiB working set, with every row verified:

| variant | MiB/s |
|---|---|
| `word` (plain stream) | 24,166 |
| `tiled` | 22,357 |
| `gemvR1` / `gemvR4` / `gemvR8` (one read serving 1/4/8 rows) | 2,691 / 2,500 / 2,594 |
| `pack4` (scalar unpack) | 2,730 |
| `pack4v` (RVV unpack) | 2,145 |
| `rvvgemv` (RVV in the vendor's shape) | 2,012 |
| vendor `i8i4_m1` (kernel probe) | 9,000 |

So the own-kernel path is technically open - the vendor uses ordinary RVV and our
kernel runs and verifies on the cluster - but it is 4.5 times slower than the kernel it
would have to beat, and no variant written here came closer than 2,700 MiB/s.
Closing that gap means matching a hand-tuned loop (`vl4r.v` segment loads,
in-register float conversion, `vfmacc` pipelining and their scheduling), which is a
re-implementation project whose payoff is bounded by the 25 % to 30 % the vendor's
four-row chunking loses in the first place. Recorded as measured, with the choice
left open rather than taken.

### A finer look corrects that: the decode kernel runs instruction families we cannot assemble

> Refuted the same day: those families are public `smt.*`, and this toolchain
> assembles them. See the correction at the end of this file.

The paragraph above said the vendor uses "ordinary RVV". Counting the instructions
objdump could not decode - it renders them `.insn 4, 0x...`, which a mnemonic
search does not see - corrects it. The library holds 494 such instructions, and
their `funct7` fields split into families:

| funct7 | library count | our toolchain |
|---|---|---|
| 0x71 | 176 | assembles: this is `smt.vmadot`, the documented family |
| 0x33 | 200 | no mnemonic, no decoder |
| 0x66 | 32 | no mnemonic, no decoder |
| 0x6B | 20 | no mnemonic, no decoder |
| 0x21 / 0x61 / 0x6A / 0x79 / 0x7A / 0x7B | 14 / 8 / 12 / 16 / 8 / 8 | no mnemonic, no decoder |

And the decode leaf itself, `ime2::gemm_kernel_i8i4_m1`, has **zero** instructions
from the 0x71 family and 40 from the other five (0x66 x12, 0x33 x10, 0x6B x8,
0x21 x6, 0x6A x4). So the kernel that produces 9.0 GiB/s runs instructions this
workspace's assembler cannot emit at all, while everything written here - `pack4`,
`pack4v`, `rvvgemv`, the vmadot probes - is from the family it does not use.

That is why the 4.5x gap is not a tuning problem to grind out: the encodings are a
prerequisite, as the IME programming model was before the specification appeared -
except this time the specification that exists documents a family the decode kernel
does not use. Recorded, with the vendor-kernel facts and the np=4 / lockstep
guidance unchanged.

### A round over the public material: the decode kernel's families are not published

> Refuted the same day: the vendor's own `docs-ai` architecture chapter documents
> them, and this toolchain decodes and assembles them. See the correction at the
> end of this file.

Checked, in order:

| source | result |
|---|---|
| the IME specification (26-line instruction list, release v1.0 PDF) | lists only `vmadot*` (funct7 0x70/0x71) and `vfmadot*` (0x74) - none of the decode kernel's five families |
| every published toolchain (`archive.spacemit.com/toolchain/`, newest v1.2.4 of this line and the 2026-06-30 musl-oh build) | identical assembler mnemonic set (34 `smt.*`), identical binutils revision, and neither decodes the decode kernel's words (40 `.insn` each) |
| the vendor's public repositories | one IME repository (the specification), documentation trees and SDK/backend repositories; nothing carrying those encodings |

So the families the 9.0 GiB/s kernel runs are **not in public material** - not in
the spec, not in any published assembler, not in any published disassembler, not
in the repositories. That is why no amount of work here closes the 4.5x gap: the
instructions cannot be assembled in this workspace, and obtaining them (or an
assembler for them) is a human step.

**One public family is still untested**: `sf.vqmacc.2x8x2` and `sf.vqmacc.4x8x4`
with their `u`/`su`/`us` forms - opcode 0x5b, quad-nibble matrix MACs, public,
assemblable here with `-march=rv64gcv_xsfvqmaccdod_xsfvqmaccqoq`, and shaped
exactly like the row counts the vendor's kernels are named after. It is the only
instruction family available in this workspace that is designed for the same job,
which makes it the one own-kernel experiment still worth running before the route
is closed.

### That family traps on the A100, which closes the route

`a100_bandwidth --sfqmac` launches one kernel per core whose only work is the
instruction in question, accumulator in the launch buffer, nothing else assumed:

| instruction | result |
|---|---|
| `sf.vqmacc.4x8x4 v28, v0, v1` | the launch never returns - no `synced`, no values, the process dies; the cluster stays healthy (`a100_probe`: `kernels=ok mismatches=0 us_per_launch=29.2`) |
| `vmadot v28, v0, v1` (same path, one instruction swapped) | completes, `[16 0 0 0 0 0 0 0]` on every core |

The control is what makes it a statement about the instruction: identical launch
path, buffers, grid and reading code; one instruction syncs, the other never does.
That is an illegal instruction on the AI cores - binutils encodes the extension,
the silicon does not implement it, and the runtime does not surface the trap.

With that, every public option on this cluster is measured:

| family | on the A100 | our best | vendor |
|---|---|---|---|
| `smt.vmadot*` (spec, funct7 0x71) | runs | 2,012-2,730 MiB/s | 9,000 (`i8i4_m1`) |
| `sf.vqmacc.*` (0x5b) | **traps** | - | - |
| plain RVV | runs | 2,730 MiB/s (`pack4`) | - |
| the decode kernel's own families (0x21/0x33/0x66/0x6A/0x6B) | runs (their code) | **public `smt.*`, assembled here byte-identically** | 9,000 |

So the family designed for this job, `sf.vqmacc`, traps on the silicon, and the RVV
kernels written here are 4.5 times behind. The third clause recorded here - that
the 9,000 MiB/s family has no public assembler - is refuted below. The A100
parallel-decode guidance is unchanged (np=4 as the knee, lockstep rather than the
server, the four-row chunking as the vendor's).

### Correction: the decode kernel's families are public `smt.*`, and this toolchain assembles them

Two sections above are wrong, for two separate reasons, and the fix is worth more
than the error was.

**The disassembler was told not to decode them.** This toolchain's RISC-V
disassembler gates each opcode behind an architecture string, which it takes from
the ELF attributes of the file it is disassembling. The vendor library advertises
`Tag_RISCV_arch: "rv64i2p1_..._zvl128b1p0"` with no vendor extension, so every one
of its vendor instructions printed as `.insn 4, 0x...`. `-M max` - "disassemble
without checking architecture string" - removes the gate, and the number of
undecoded words goes to **zero**:

| mnemonic | count in `libggml-cpu.so.0.16.0` |
|---|---|
| `smt.vpack.vv` | 150 |
| `smt.vmadot` | 144 |
| `smt.vupack.vv` | 50 |
| `smt.vmadotsu.hp` | 48 |
| `smt.vmadotu.hp` | 32 |
| `smt.vmadotsu` | 28 |
| `smt.vmadot.hp` | 16 |
| `smt.vnpack4.vv` | 14 |
| `smt.vmadotu` | 12 |

494 is exactly the number that had been counted as undecodable, so the five
"families" tabulated by `funct7` were this one public instruction set, unread.

**The assembler test used the wrong operand form.** The `.hp` forms take five
operands plus a data-type suffix (`vd, vs1, vs2, v0/v1, imm3, i4|i8`), the packing
forms four (`vd, vs1, vs2, imm2`); the earlier check tried three. Written the way
the vendor's own disassembly prints them, this toolchain emits the vendor's words
byte for byte:

| the vendor's leaf | this toolchain |
|---|---|
| `d645082b` `smt.vmadotsu.hp v16,v10,v4,v1,0,i4` | identical |
| `cc44082b` `smt.vmadotu.hp v16,v8,v4,v0,0,i4` | identical |
| `67281c2b` `smt.vpack.vv v24,v16,v18,1` | identical |
| `4231b42b` `smt.vnpack4.vv v8,v3,v3,3` | identical |

And the third reason, the one that should have stopped the earlier round before it
started: the vendor's public `docs-ai` tree documents the whole family, and has been
cloned in this workspace since 2026-09-17. `en/architecture/ime_extension.md` is the
`Zvvm_spacemit` profile, version 0.6, "Status: Public Release", 2026-04-13; chapter
5.4 defines `smt.vmadot.hp*` as the block-quantization path `C <- (A x B) x S + C`
with FP16/BF16 scales, chapter 8.4 gives formal encodings whose fields reproduce
`0xd645082b` by hand, and the register constraint the assembler enforces (`v0`/`v1`)
is the specification's own.

That makes the vendor's decode kernel readable as what it is: `vnpack4` packs the
4-bit weights, `vmadotsu.hp`/`vmadotu.hp` multiply Int4 and Int8 tiles against FP16
block scales and accumulate into FP16, and `vpack` interleaves the accumulators.
The 4.5x gap is now a re-implementation task with the right instructions in hand,
not a route that waits on vendor material. `sf.vqmacc.*` still traps on this silicon
and stays out. The next measurement is the one below.

### The vendor's family runs on the A100

`a100_bandwidth --smthp` launches one kernel per core whose entire work is
`smt.vmadotsu.hp v24,v16,v16,v0,0,i4`, operands and accumulator in the launch
buffer. Uniform inputs make the answer independent of the tile extent: every
nibble of both operands is 1 and the scale register holds FP16 1.0 in every lane,
so each output lane is 32 products of 1x1 times 1.0 = 32.0 = `0x5000`.

| kernel | per core |
|---|---|
| `--smthpctl`, the control: same launch and buffers, no matrix instruction | its own tile id written and read back per core |
| `--smthp` | **`0x5000` in all 64 accumulator lanes on all eight cores** |

So the family the 9,000 MiB/s decode kernel is built from executes on this
silicon, returns the value the published semantics predict, fills the whole tile,
and leaves the core able to store afterwards.

### The vector length was being read in the wrong unit

`BackendInfo::vlen` is documented in `spert.hpp` as bytes, and earlier rounds in
this project read its `128` as bits and reasoned about 16-byte vector registers.
Measured inside a kernel on the AI cores, `csrr vlenb` returns **128** on all eight
cores, and the vendor's own instruction-set document fixes the A100 for this
family at `VLEN = 1024 bits`, `lambda = 8`, `MUL_C = 1`, `SEW = 16`. So a register
is 128 bytes, 8x32x8 Int4 is one register per operand, and the 8x8 accumulator is
one register of 64 FP16 lanes. The X100 backend reporting `vlen=32` (32 bytes =
256 bits, the A60-class core) is the second check on the unit. Every earlier
statement here that turned the A100's tile into "two 16-byte registers" or a
"broadcast across registers" is withdrawn; the 64-lane result above is what the
geometry actually produces.

`--smthpctl` is not decoration: the first version of this probe reported a scatter
that looked like a property of the instruction, and the control produced the same
scatter with no matrix instruction in the kernel at all. Three probe defects came
out of it in order - a 128-byte store against a 256-byte window, and twice an
inline-asm output register aliasing an input pointer, once sending the stores to
address 128 and once writing a tile id of `0x0080`. Any per-core number in this
project's probe style now has to pass an identity-writing control launch of the
same shape first: `a100_probe`'s `out[tile]` convention and the layout masks do,
and the `--sfqmac` control's per-core line is re-opened as unverified - its
conclusion, that the family never syncs, is untouched.

### The vendor's own shape streams 22.7 GiB/s, against the vendor kernel's 9.0

`a100_bandwidth` now carries that shape as a variant: one `smt.vmadotsu.hp` per
128-byte weight register plus the 16-byte load of the eight FP16 column scales -
144 bytes of traffic per instruction, the same accounting the vendor's `i8i4_m1`
entry point is charged for its 9.0 GiB/s. The loads are checked exactly (every
weight register XORed into a 16-lane accumulator, reproduced on the host), and the
instruction's own arithmetic is checked by `--smthp` above.

| kernel | 64 MiB | 256 MiB |
|---|---|---|
| `word` (plain stream, the roof) | 25,176 | 24,623 |
| `tiled` (the decode path's own walk) | 22,710 | 23,475 |
| `smthp` (vendor shape) | **22,679** | **22,670** |
| `smthpnd` (same loop, no matrix instruction) | 22,678 | 21,525 |
| `pack4` (scalar unpack) | 2,957 | 2,669 |
| `gemvR1` (one read serving one row) | 2,714 | 2,570 |
| `rvvgemv` (our RVV gemv) | 2,094 | 2,008 |
| vendor `i8i4_m1` (kernel probe, earlier round) | 9,000 | - |

All MiB/s, all with the same data-plus-scales accounting. Three things follow.
The instruction is free at this granularity - `smthp` and `smthpnd` are within
noise and both sit at the `tiled` walk. An own kernel in the vendor's own shape
streams 2.5x the vendor's one-row kernel and 8-11x our best RVV gemv. And
single-sequence decode is already at this roof in aggregate: 1,465-1,475 MiB per
token at 14.0-14.4 t/s is 21.1 GiB/s against the 22.7 measured here, so the
vendor's 9.0 GiB/s is a per-kernel figure, not the system's ceiling.

The case that is *not* at the roof is multi-row: at np=4 the run moves 402 MiB per
token in aggregate at 66.9 % of the bus, on the vendor's four-row kernel, measured
earlier at 5.8 GiB/s. 402 MiB per token against this stream is 56 tokens/s
aggregate against the 41.0 measured today, which is the next experiment.

One caution about the harness, recorded rather than hidden: the submitting process
dies at exit flakily, after all work is done and after every checksum has passed -
on this revision and on the one before the matrix kernels were added, with and
without the matrix instruction, and `stdbuf -o0` does not change it. A run is
therefore accepted on: every `RESULT` line present, every `checksum=ok`, and
`a100_probe` healthy either side. The process exit status is not a usable success
signal here until that defect is found.

### The decode kernel is source, not a black box

`spacemit_kernels::ime2::gemm_kernel_i8i4_m1` is in the pinned fork, not only in the
prebuilt library: `ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp` of
`projects/llama.cpp/src/llama.cpp`, dispatched by `gemm_kernel_i8i4` -
`count_m >= 4` takes the `_m4` kernel, everything else `_m1`. Its own comments give
the two formats:

| side | per k-block | bytes |
|---|---|---|
| A, activations, 1 row, int8 | `fp32 scale | int16 asum | 32 x int8` | 38 |
| B, weights, 32 columns, int4 | `32 x fp16 scale | 32 columns x 16 bytes` | 576 |

and its inner loop, per k-block: `vl4r.v` (512 bytes of weights), `vle8.v` (64 bytes
of scales), `vle8.v` (32 bytes of activations), two `vnpack4.vv`, eight MACs
(`vmadotsu.hp` / `vmadotu.hp`), three `vpack.vv`, `vfwmul.vv`, `vfmacc.vf` - about
two dozen vector instructions for 614 bytes, accumulating in fp32 in `v2`.

Two things follow. The family measured above is the family this kernel runs, so the
2.4x between 9.0 and 22.7 GiB/s is not instruction availability, not the format and
not the memory path; and the probe's byte accounting is confirmed correct for it
(18 bytes per column per block is exactly the B side of that table). What is left
as the cause is the inner loop rather than the calls, and that is measured, not
argued: a census in the kernels (`projects/llama.cpp` patches/0001-ime2-call-shape-census.patch)
gives every `i8i4_m1` call a full 32-column group and 71.35 k-blocks on average -
about 40 KiB of weights per call, so per-call overhead is refuted. Per k-block the
loop issues roughly two dozen vector instructions for 614 bytes, eight of them
hp-MACs into four accumulators followed by three packs and two conversions into the
fp32 FMA. That was the last hypothesis standing for the 2.4x, and the ladder in
`projects/llama.cpp` has since retired it: emptying that loop completely - no loads,
no arithmetic, only the counter - leaves decode at 14.47 t/s against 14.46 for the
full kernel. The kernel is not on decode's critical path, which is the same statement
as point 3 above from the other direction: the system is already at 21.2 of the
22.7 GiB/s it can stream. The build loop that produced that result exists and is
cheap: `projects/llama.cpp/scripts/build.sh` rebuilds the stage in about ten seconds
incrementally, and the staged library runs under the board's own `llama-bench` at
124.02 PP128 / 14.46 TG128 against the packaged library's 104.92 / 13.96.

## Workspace entry

```xml
<project name="spine-runtime.git" path="projects/spine-runtime" />
```
