# Instrumentation reference

The `mandelHybrid` binary is heavily instrumented for profiling and for
visualising the load-balancing decomposition. This document describes each
mechanism, what it measures, the exact stderr line it emits, and how to consume
its output. All of it operates at near-zero overhead when no profiler is
attached and `quiet`/`viz` are off.

For results derived from this instrumentation see the reports under
[`reports/`](reports/) and the raw data under
[`experiments/`](experiments/). For the architecture and the full code
walkthrough see [`reports/14-architecture-guide.tex`](reports/14-architecture-guide.tex).

---

## Running the binary

```bash
./mandelHybrid spec.in <numThreads> <gpuEnable> [diffThreshold] [pixelThreshold] [quiet] [save] [viz] [vizFrame]
```

| # | arg | default | meaning |
|---|---|---|---|
| 2 | `numThreads` | core count | total CPU+GPU worker threads (thread 0 drives the GPU when `gpuEnable=1`) |
| 3 | `gpuEnable` | 1 | 1 enables GPU; 0 is CPU-only (and flips the work queue to LPT — see below). On a `make GPU=0` build (CPU-only, `kernel_stub.cpp`, no CUDA needed) only 0 is valid — the stub aborts loudly if the GPU path is invoked |
| 4 | `diffThreshold` | 0.5 (legacy) | region is "uniform" when `(maxIter − minIter) < diffT × maxIter`. **Recommended default: 0.1** (report 06) |
| 5 | `pixelThreshold` | 32768 | regions smaller than this many pixels are computed without further splitting |
| 6 | `quiet` | 0 | 1 suppresses the per-region `[CPU region]` / `[GPU region]` / `[… metrics]` lines; per-thread, GPU, and `[OUTLIER]` summaries still emit |
| 7 | `save` | 1 | 0 skips `img->save()` so pure compute can be timed (forced to 0 when `viz≥2`) |
| 8 | `viz` | 0 | subdivision visualizer: `1` = single-frame depth animation, `2` = full-run partition overlay, `3` = full-run split-by-split process. See [Visualization modes](#5-visualization-modes-mainccpp-mandelframe). |
| 9 | `vizFrame` | 0 | for `viz=1` only: which interpolated frame index to freeze on |

Example: `./mandelHybrid spec.in 12 1 0.1 32768 1 1`

`spec.in` format:

```
numframes resolutionX resolutionY imageFilePrefix
upperCornerX upperCornerY lowerCornerX lowerCornerY maxIterations  # first frame
upperCornerX upperCornerY lowerCornerX lowerCornerY maxIterations  # last frame
```

Intermediate frames are interpolated linearly between the two corner specs.
PNG output is written to the current working directory using the prefix
(e.g. `img` → `img0000.png`, `img0001.png`, …). To redirect into a
subdirectory, include a trailing slash in the prefix (`img/`) and `mkdir -p`
first.

> **zsh note.** Under zsh an unquoted shell variable holding multiple words is
> **not** word-split, so `./mandelHybrid spec.in $CFG` (with `CFG="12 1 0.1"`)
> passes the whole string as `argv[2]` and silently runs the defaults. Pass run
> arguments as literal tokens and confirm them against the `[config]` banner
> (below) before trusting any timing.

---

## Running on one node or many (`DIST_*` frame distribution)

With no `DIST_*` environment variables set, the binary renders **every**
frame — the classic single-node run; nothing below applies. Setting them
turns the same binary into one *rank* of a multi-node run: frames are
distributed **across** nodes while the region-level work queue stays
**intra-node**, untouched. Every rank reads the *same* `spec.in` (the corner
interpolation still accumulates over all frame indices, so an owned frame's
corners are bit-identical to a single-node run); non-owned frames get no
`MandelFrame` and no root region, so per-node frame-buffer memory drops to the
owned share. Output PNGs keep the **global** frame index, which makes
multi-node collection a plain union — no renumbering, no collisions.

### Ownership modes (environment variables)

Precedence: `DIST_FRAMES` > `DIST_WEIGHTS` > block-cyclic.

| Variable | Mode | Meaning |
|---|---|---|
| `DIST_NODES`, `DIST_RANK`, `DIST_BLOCK` | block-cyclic (static) | frame `i` is owned iff `(i / DIST_BLOCK) % DIST_NODES == DIST_RANK`. `DIST_BLOCK=1` = pure cyclic (default); `k` = block-cyclic; `≥ ceil(frames/N)` = contiguous block |
| `DIST_WEIGHTS` (+ `DIST_SEED`, default 1234) | weighted-random (static) | `"1,4"` = one positive integer per rank; frame `i` belongs to the rank whose cumulative-weight interval contains `splitmix64(seed+i) mod W`. Pure uint64 arithmetic → every rank on every machine computes the same partition with **no communication**; rank `r` owns ~`F·w_r/W` frames randomly interleaved along the zoom's cost trend. `DIST_BLOCK` is ignored |
| `DIST_FRAMES` | explicit list | `"3,17,42"` = render exactly these global frames. The dispenser interface: `dist_dynamic.sh` dispatches every chunk this way. All other `DIST_*` are ignored |

Malformed settings exit non-zero before any compute (`Bad DIST_*
environment`, `DIST_WEIGHTS has k entries…`, `DIST_FRAMES index out of
range`). `viz=1` freezes a single frame, so frame distribution is ignored
there with a warning. **Weights rule** (report 28): a node's share should be
proportional to `1/(p_i + l_i)` — per-frame compute *plus* per-frame
collection cost — not `1/p_i` alone; with batched (tar) collection the two
coincide within noise.

### The `[dist]` banner

Each rank prints one stderr line right after the `[config]` banner — read
both before trusting a distributed run (there are no dist fields in
`[config]` itself):

```
[dist] rank=0 nodes=3 block=1 ownedFrames=34 of 100          # block-cyclic
[dist] mode=weighted rank=1 nodes=2 weights=1,4 seed=1234 ownedFrames=79 of 100
[dist] mode=list ownedFrames=12 of 100                       # DIST_FRAMES
[dist] viz=1 freezes a single frame; ignoring DIST_*
```

`ownedFrames` across all ranks must sum to the frame count — the union check
the benchmark scripts perform after collection.

### Orchestrators

Both scripts drive N ranks over SSH from one coordinator (GNU parallel; no
node-to-node connections, no resident daemon) and share the hosts-file
format — one node per line, rank `r` = line `r+1`:

```
host [rdir [args...]]
```

`host` = SSH destination, or `":"` for this machine without SSH; `rdir` =
that host's project root (`-`/omitted = the `RDIR` default); `args` = that
host's `mandelHybrid` arguments — which is how heterogeneous nodes join, e.g.
a CPU-only `make GPU=0` node runs `8 0 0.1 32768 1 1` while a hybrid node
runs `20 1 0.1 32768 1 1`. Binaries must already be built on each node; the
scripts ship only the spec and collect only PNGs. Each rank runs in
`rdir/dist_work/rank<r>` (the spec's short prefix is reused — do **not**
encode per-node paths into the prefix; `fin >> imageFilePrefix` overflows
`char[42]` on prefixes > 41 chars).

- **`scripts/dist_frames.sh hosts.txt`** — static distribution. Env knobs:
  `SPEC`, `OUT` (collection dir), `RDIR`, `ARGS` (defaults for hosts-file
  omissions), `BLOCK` (= `DIST_BLOCK`), `WEIGHTS`/`SEED` (= `DIST_WEIGHTS`/
  `DIST_SEED`; when set, `BLOCK` is ignored), `JOBS` (concurrent node jobs;
  the local identity check uses `JOBS=1` because simulated ranks share one
  GPU).
- **`scripts/dist_dynamic.sh hosts.txt`** — weighted-random initial bags +
  coordinator-driven **work stealing**: per-node drivers dispatch guided
  chunks (half the remaining bag, min `KMIN`, capped at `KCAP·w_r` frames)
  as repeated `DIST_FRAMES` invocations; a node whose bag empties steals half
  the richest node's undispatched tail (bag mutations under one `flock`;
  in-flight chunks are already out of the bags, so every frame renders
  exactly once). Env knobs: `SPEC`/`OUT`/`RDIR`/`ARGS` as above, `WEIGHTS`
  (default all-1), `SEED` (default 1234), `KMIN` (default 4), `KCAP`
  (default 8). The weight-proportional cap bounds every node's unstealable
  in-flight exposure at the same wall-time budget (report 26); note each
  chunk is a fresh process — a GPU node pays a CUDA-context creation per
  chunk, which is the dynamic mode's +21 % premium over weighted static when
  weights are already right (report 27).

Collection in both scripts is a **single tar stream per rank**
(`ssh host "cd wd && tar cf - prefix*.png" | tar xmf -`), not per-file scp —
the SCP/SFTP protocol pays 2–3 round trips *per file* (~68 ms/file at 25 ms
RTT, 6.60 s vs 0.96 s on a 79-file/36 MB share; report 27).

### Verifying a distributed setup

`experiments/25-frame-distribution/verify_dist.sh` is the byte-identity
check: 3 simulated local ranks vs. a single-process reference, executor-pinned
(CPU-only and GPU-only separately, cyclic and block-cyclic). Pinning matters:
CPU-vs-CUDA FP64 last-ulp rounding makes *hybrid* output timing-dependent, and
hybrid output across different GPU generations/nvcc versions (sm_75/13.2 vs
sm_89/13.3) is not expected to be byte-identical either. Cross-machine
*CPU* identity holds because the build uses plain `-O2` — no `-march`, no FMA
contraction (verified Coffee Lake g++-15 vs Haswell g++ 16, report 25);
adding `-march=native` would break it.

---

## What gets instrumented

### 1. NVTX ranges  (`kernel.cu`, `mandelregion.cpp`)

NVTX (NVIDIA Tools Extension) is a CPU-side annotation API.
`nvtxRangePush("label")` / `nvtxRangePop()` bracket a named interval on the
calling thread's stack. Nsight Systems records the open/close timestamps and
renders them as coloured bars in the CPU timeline, one lane per thread.

NVTX3 (CUDA 11+) is header-only: it `dlopen()`s the profiler's runtime
library only when Nsight Systems is active. When the program runs standalone
there is no measurable overhead. The Makefile links `-ldl` instead of an
explicit `-lnvToolsExt`.

Three ranges are emitted:

- **`examine region`** — `mandelregion.cpp`, `MandelRegion::examine`. Wraps
  the full decision cycle for one region: the 9-point uniformity sample plus
  either a pixel compute or a four-way split back onto the work queue. Lets you
  visually count how many `examine()` calls each thread handles and compare
  depth/duration distributions between CPU and GPU workers.
- **`CPU region compute`** — `mandelregion.cpp`, `MandelRegion::compute`
  (CPU branch). Wraps the pixel iteration loop for CPU-computed regions.
- **`GPU region compute`** — `kernel.cu`, `hostFE`. Wraps the full GPU path:
  kernel launch + device-to-host memcpy + result copy into the QImage buffer.
  Appears on thread 0's lane alongside the automatically recorded CUDA API
  calls.

The `examine region − (CPU/GPU region compute)` difference on the timeline is
the sampling/dispatch overhead; report 09 measured it at +29 µs per `examine()`
(negligible against the iteration loop).

### 2. CUDA event timing  (`kernel.cu`)

`cudaEvent_t` is an opaque GPU-side timestamp. `cudaEventRecord()` inserts a
"record this timestamp" command into the CUDA stream; it completes
asynchronously alongside kernel/memcpy work. `cudaEventSynchronize()` blocks
the host until the event has been stamped, then `cudaEventElapsedTime()`
returns wall-clock milliseconds between two events as measured by the GPU's
own hardware counter.

This is more accurate than wrapping the launch with a CPU clock because the
GPU executes asynchronously; a host-side timer would include driver/queue
overhead and could miss the true execution time.

Four events are created once in `CUDAmemSetup()` and reused across every
kernel call: `evKernelStart`, `evKernelStop`, `evMemcpyStart`, `evMemcpyStop`.

Each call to `hostFE()` prints a per-region line to stderr (unless
`quiet=1`):

```
[GPU region   42] size  960x 540  kernel 1.234 ms  D→H 0.456 ms
```

`CUDAmemCleanup()` prints an aggregate summary at shutdown:

```
[GPU profiling summary]
  Regions computed on GPU : 3679
  Total kernel time       : 21340.000 ms  (avg 5.800 ms/region)
  Total D→H memcpy time   : 88.300 ms  (avg 0.024 ms/region)
```

**`cudaMemcpy` time is dominated by kernel-wait, not transfer.** The `D→H`
event pair brackets the synchronous `cudaMemcpy`, which implicitly waits for the
kernel to finish; the actual PCIe transfer is ~15–24 µs/region (~95 ms total
for a 100-frame run), so ~99 % of any `cudaMemcpy` duration reported by Nsight
is kernel-wait. PCIe bandwidth is *not* a bottleneck (report 01).

### 3. CPU thread timing  (`mandelregion.cpp`)

`clock_gettime(CLOCK_MONOTONIC)` brackets the pixel iteration loop for every
CPU-computed region. `thread_local` counters accumulate region count and total
compute time per thread with no synchronization overhead.

When a thread's work queue drains, `MandelRegion::printCPUSummary()` prints:

```
[CPU profiling summary - this thread]
  Regions computed on CPU : 467
  Total compute time      : 49210.000 ms  (avg 105.376 ms/region)
```

The GPU thread (thread 0) prints nothing from `printCPUSummary()` because its
regions are counted inside `hostFE()` and summarised in `CUDAmemCleanup()`
(the per-thread `cpuRegionCount` is 0, so the function returns early).

#### Why three timers (and not one)

CPU regions use `clock_gettime`, GPU regions use CUDA events (§2), and NVTX (§1)
labels both for Nsight Systems. Three tools by design — each measures a different
thing and is consumed differently:

- **`clock_gettime` (CPU compute)** is *self-contained and always on*: it emits
  the machine-parseable stderr summaries with **no profiler attached and
  near-zero overhead** (`thread_local` counters). It is the source of truth for
  every number in the reports — the per-thread totals, the conditional
  `[OUTLIER]` dump (application logic a passive trace cannot do: "fire when a
  region exceeds 5 s and print its geometry/`cardioidBulb`/inset"), and the
  `[total_elapsed_s]` wall that `sweep_fig1115.sh` greps across 21+ batch runs.
  Those sweeps could not run under a profiler.
- **CUDA events (GPU compute)** are required *because the GPU is asynchronous*: a
  host clock around the launch would time the driver/queue, not the kernel. The
  events are stamped on the GPU's own hardware counter.
- **NVTX (labels)** measures *nothing on its own* — `nvtxRangePush/Pop` are
  near-no-ops until Nsight Systems attaches. They give the one thing scalars
  cannot: the **timeline** — per-thread lanes, examine/compute/GPU overlap, GPU
  idle at the tail, and correlation with the CUDA API/kernel rows on a shared
  clock.

So `clock_gettime` answers *how long / how many* from any plain run; NVTX answers
*when / where / overlapping with what*, but only under `nsys`. Using NVTX alone
would couple every measurement to a profiling session (and lose the conditional
outlier dump and the directly-greppable wall); using `clock_gettime` alone would
lose the cross-thread timeline and the automatic CUDA correlation. Complementary,
not redundant — and `ncu` (kernel deep-dive, below) adds a fourth layer:
*in-kernel* hardware counters neither of the others can reach.

### 4. Region work metrics  (`mandelregion.cpp`)

Both compute branches accumulate two work metrics in the same pass that copies
results into the QImage, reusing the existing `color == MAXGRAY` branch so the
cost is one add per pixel (negligible against the `diverge()` loop):

- **`meanIter`** — mean iteration count over the region's pixels (`iterSum / npix`).
- **`inset`** — fraction of pixels that reached `MAXITER`, i.e. are inside the
  set (`insetCount / npix`). A high interior fraction means a warp-coherent,
  GPU-friendly region; the binding outliers are 84–97 % interior.

Per-region lines (unless `quiet=1`) — note both now carry `frame` and `depth`:

```
[CPU region  467] frame  73 depth 1 size  960x 540  compute 13292.0 ms  meanIter 9998.7  inset 97.4%
[GPU region metrics] frame  73 depth 1 size  960x 540  meanIter 9998.7  inset 97.4%
```

(The GPU thread emits *two* lines per region: the `[GPU region NNN]` timing line
from `hostFE()` in §2 and the `[GPU region metrics]` line from `compute()` here.)

### 5. Outlier dump + cardioid/bulb test  (`mandelregion.cpp`)

Any CPU region whose compute time exceeds `OUTLIER_MS` (5000 ms) is dumped in
full **regardless of the quiet flag** — these are the load-balancing outliers
worth dissecting:

```
[OUTLIER] frame=73 depth=1 img=(0,0) px=960x540 c=[-0.743644,0.131826]..[-0.743021,0.131474] corners=10000/10000/10000/10000 spread=0 meanIter=9998.7 inset=97.4% cardioidBulb=0 ms=13292.0
```

Fields: frame index, recursion depth, top-left pixel coordinate, pixel
dimensions, the complex-plane rectangle, the four corner iteration counts, their
spread (`max − min`), the work metrics from §4, and `cardioidBulb` — the result
of `MandelRegion::inMainCardioidOrBulb()` applied to all four corners.

`inMainCardioidOrBulb(x,y)` is the exact algebraic membership test for the two
largest interior components of the Mandelbrot set:

- main cardioid: `q = (x−¼)² + y²; q(q + (x−¼)) ≤ ¼y²`
- period-2 bulb: `(x+1)² + y² ≤ 1/16`

A point passing it is provably in the set, so a region whose four corners all
pass would be a candidate for a constant-fill interior certificate. The binding
outliers report `cardioidBulb=0` (they are *minibrot* interiors, not the main
cardioid/bulb), which is why a cardioid certificate cannot relieve them — the
diagnostic that motivated the work-reduction roadmap in `CLAUDE.md`.

### 6. End-to-end wall timer + config banner  (`main.cpp`)

`clock_gettime(CLOCK_MONOTONIC)` brackets the compute phase from
`CUDAmemSetup` (when GPU is enabled) through every worker thread's `wait()`
return. Image saves are inside the bracket because they happen on whichever CPU
worker decrements `remainingRegions` to 0. The visualization passes (§ below)
run *after* the timer stops, so they never perturb the measurement. One
machine-parseable line is printed at shutdown:

```
[total_elapsed_s] 49.861203
```

`scripts/sweep_fig1115.sh` greps for this exact prefix to populate its CSV.
`CLOCK_MONOTONIC` is used (rather than `CLOCK_REALTIME`) so NTP/DST jumps
cannot corrupt the measurement.

A self-describing configuration banner is printed at startup so each log file
identifies its own run (note the `viz`/`vizFrame` fields, added with the
visualizer):

```
[config] numThr=12 gpuEnable=1 diffT=0.100 pixT=32768 res=1920x1080 frames=100 quiet=1 save=0 viz=0 vizFrame=-1
```

`vizFrame` is reported as `-1` when `viz=0`. **Always read this banner first** —
it is the ground truth for what the run actually did.

---

## The work queue: GPU affinity (what the profile *shows*)

`WorkQueue` (`workqueue.cpp/.h`) is a mutex-protected min-max `std::multiset`
ordered by pixel count (`MandelRegion::Compare`, ascending). `extract(bool isGPU)`
is **executor-aware**:

- the **GPU thread** (`isGPU=true`) pops the **largest** pending region
  (`--queue.end()`) — it is ~30× faster on big coherent interior regions;
- **CPU threads** pop the **smallest** (`queue.begin()`);
- a **guard**: when there is no GPU in the run (`gpuPresent=false`, set from
  `gpuEnable`), CPU threads pop the *largest* too — largest-first / LPT — so big
  regions are not deferred to the tail.

This is why the `[GPU region metrics]` and `[CPU region]` lines show the big
`960×540`/`480×270` interior regions landing on the GPU and the small divergent
ones on the CPU pool. It is the −3.6 % win of report 12; the full rationale and
the negative shared-priority-queue result it supersedes (report 11) are in
`CLAUDE.md`.

---

## Visualization modes  (`main.cpp`, `mandelframe.*`)

When `viz≠0`, every examined region registers a `VizRect` (pixel rectangle,
depth, leaf/split flag, executor) with its owner frame under a mutex
(`MandelFrame::addVizRect`). Recording is gated on `vizMode` so it never touches
a timed run. After the wall timer stops, a shared `drawVizOverlay()` strokes the
recorded outlines **2 px inward** of each region's true boundary:

- grey `(90,90,90)` — internal/split nodes (the subdivision skeleton);
- cyan `(0,255,255)` — leaves computed on the **GPU** thread;
- yellow `(255,230,0)` — leaves computed on a **CPU** thread.

Growing the 2 px stroke inward (an outer rect on the boundary plus an inner rect
inset 1 px) keeps each colour entirely inside its own region, so a GPU/CPU
neighbour shows two solid bands rather than two abutting 1 px lines that blend to
green when viewed at scale.

| `viz` | function | output | builds |
|---|---|---|---|
| 1 | `generateDepthFrames` | `<prefix>_fNNNN_dKK.png` (one PNG per depth `K` of frame `NNNN`) | 1 frame (`vizFrame`) |
| 2 | `generateSequenceFrames` | `<prefix>NNNN.png` (each frame overlaid with its full partition) | all frames |
| 3 | `generateProcessFrames` | `<prefix>NNNNN.png` (per frame, one PNG per depth, then advance the camera; single global counter) | all frames |

`viz≥2` forces `save=0` (the overlaid PNGs are the deliverable). `viz=3` is a
deterministic depth-ordered reconstruction, not a wall-clock replay (the queue is
processed concurrently). The decomposition is byte-identical with and without
`viz` — the visualizer is instrumentation only (verified in report 08).

Assemble to video at native resolution with **no temporal interpolation** so the
overlay lines stay aligned to their frame:

```bash
ffmpeg -framerate 12 -i img%05d.png -bf 0 -fps_mode passthrough \
       -pix_fmt yuv444p out.mp4        # yuv444p = zero chroma blending
```

---

## Batch profiling helpers

### `scripts/sweep_fig1115.sh`

Runs seven `(numThreads, gpuEnable)` configurations N times each and emits
one CSV row per run, grepping the `[total_elapsed_s]` line from each run's
stderr. Auto-resolves the project root from its own path, so it works from
any cwd.

```bash
scripts/sweep_fig1115.sh                                        # 3 reps, OUT=experiments/sweep_results
REPS=5 OUT=experiments/05-fig1115-postfix scripts/sweep_fig1115.sh
DIFFT=0.1 scripts/sweep_fig1115.sh                              # recommended diffT
SAVE=0 scripts/sweep_fig1115.sh                                 # pure-compute timing (skip PNGs)
```

Each per-config run uses its own scratch `cwd` so the PNG output for one rep
doesn't pollute the next, and the scratch is cleaned between runs to keep
disk usage bounded (~50 MB peak for a single full-HD 100-frame render).

### `scripts/plot_fig1115.py`

Reads a sweep CSV, computes mean and standard deviation per configuration,
and emits a Fig 11.15-style chart (blue bars with error caps, red
percent-of-GPU-alone line on the right axis).

```bash
scripts/plot_fig1115.py experiments/05-fig1115-postfix/results.csv \
                        experiments/05-fig1115-postfix/fig1115.png
```

### `scripts/build_report.sh`

Runs `pdflatex` twice (so cross-references resolve) for any report in
`reports/`, or for all of them:

```bash
scripts/build_report.sh 03-bug-analysis
scripts/build_report.sh all
```

---

## How to profile with Nsight

### Nsight Systems (timeline view, recommended first step)

```bash
mkdir -p experiments/my-profile
cd experiments/my-profile
nsys profile --trace=cuda,nvtx,osrt -o report ../../mandelHybrid ../../spec.in
nsys-ui report.nsys-rep            # GUI timeline
nsys stats report.nsys-rep         # text summary
```

For machine-readable per-range / per-kernel tables:

```bash
nsys stats --report nvtx_pushpop_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum report.nsys-rep
```

What to look for in the timeline:

- NVTX rows show `examine region`, `CPU region compute`, and
  `GPU region compute` bars side by side with CUDA API calls.
- Compare how much wall time each `CalcThr` lane spends on GPU vs CPU work.
- GPU idle time at the end of the run indicates CPU threads are still
  finishing the tail; quantifies the imbalance.
- The CUDA row shows kernel + memcpy back-to-back; a gap between them would
  indicate CPU-side stalls feeding the queue.
- `cudaMemcpy` reports its full wait time including the implicit stream
  synchronization — most of its reported duration is kernel-wait, not actual
  transfer. See report 01 for the breakdown.

### Nsight Compute (kernel deep-dive)

`ncu` reads *in-kernel* hardware counters — warp execution efficiency, FP64 pipe
utilisation, the compute/memory roofline, occupancy, warp stall reasons — that
`nsys` (a timeline tracer) cannot see. These counters are **admin-restricted**
(`ERR_NVGPUCTRPERM` for a normal user), so **capture** must run as root;
**parsing** a saved report does not. That split is wired up in
`experiments/16-ncu-divergence/`:

```bash
sudo experiments/16-ncu-divergence/capture.sh                    # capture (root) -> ncu/*.ncu-rep
ncu --import experiments/16-ncu-divergence/ncu/interior_full.ncu-rep \
    --page raw --csv                                             # parse (unprivileged)
```

`capture.sh` runs the binary **GPU-only** (`numThreads=1`) so the single GPU
thread is forced to process *every* region type — in a normal hybrid run the
affinity queue routes boundary regions to the CPU, so the GPU never sees the
divergent ones; to *measure* divergence they must be forced onto it. `ncu`
replays each kernel once per counter group, so cap the kernel count with
`--launch-count N` — a full run is otherwise very slow.

What report 16 measured (`reports/16-ncu-divergence.tex`):

- **Warp execution efficiency** — coherent interior/exterior regions run at
  *exactly* 100 % (zero divergence); divergence (down to 34 %) appears only in
  the 240×135 **pixel-floor** boundary leaves, which affinity keeps on the CPU.
  The uniformity test doubles as a coherence filter, so the GPU's actual work is
  divergence-free — **divergence is a non-lever**.
- **FP64 ceiling** — the kernel is FP64-bound at **85.6 % of peak** (~110 of
  129 GFLOPS at 1.34 GHz; FP64 is 1:32 of FP32 on the consumer-Turing TU116),
  at ~99 % occupancy with ~0 % memory traffic. This corrects report 11's ~40 %
  estimate; it is why coherent interior regions are the GPU's strength.
- **Stall reason** — `long_scoreboard` dominates (~95 %) in *every* regime,
  uniform across content: the serial `z²+c` recurrence's latency (each iterate
  depends on the previous), not memory and not divergence. The 15 % gap below
  the FP64 peak is that recurrence latency, not occupancy.

---

## Measurement hygiene

- **AC power is mandatory for timing.** On battery the CPU throttles ~3× and the
  Max-Q GPU is power-capped (a v5-era 52 s run ballooned to ~147 s and the
  GPU/CPU split inverted). The benchmark drivers (`experiments/23-…/ab.sh`,
  `experiments/25-…/bench_*.sh`) refuse to run off AC.
- **Same-day in-batch baselines.** The absolute hybrid wall drifts with ambient
  conditions (v7 baseline measured anywhere from 18.7 to 23.7 s across report
  days); every A/B in the reports therefore carries its own same-day baseline,
  and cross-day walls are never compared directly.
- **Read the `[config]` banner** of every log before quoting its numbers (zsh
  word-split caveat above) — and the `[dist]` banner too on distributed runs.
- Wall comparisons in the reports use alternating A/B reps and quote the range,
  not just the mean, so signal can be distinguished from noise.

---

## Existing measurement data

| Directory | Binary | Contents |
|---|---|---|
| `experiments/01-initial-profile/` | `binary-v0-buggy` (`0f782fd`) | nsys profile of the original code |
| `experiments/02-fig1115-prefix/`  | `binary-v0-buggy` (`0f782fd`) | Fig 11.15 sweep CSV + per-rep stderr logs |
| `experiments/04-postfix-profile/` | `binary-v1-bugfix` (`d5bf30c`) | nsys profile of the post-fix code |
| `experiments/05-fig1115-postfix/` | `binary-v1-bugfix` (`d5bf30c`) | Fig 11.15 sweep CSV + per-rep stderr logs |
| `experiments/06-difft-postfix/`   | `binary-v1-bugfix` (`d5bf30c`) | `diffThreshold` sweep + `quiet0/` max-region runs |
| `experiments/07-difft-prefix/`    | `binary-v0-buggy` (`0f782fd`) | `diffThreshold` sweep + `quiet0/` max-region runs |
| `experiments/08-region-metrics-viz/` | `binary-v2-viz` (`616c147`) | viz depth frames + montage/GIF, metrics + `[OUTLIER]` logs, A/B perf vs `d5bf30c` |
| `experiments/09-9point-sampling/` | `binary-v3-9point` (`ab47540`) | 9pt vs 4pt metrics, nsys NVTX/CUDA stats, A/B perf, frame-89 viz |
| `experiments/10-difft-9point/`    | `binary-v3-9point` (`ab47540`) | `diffThreshold` re-sweep confirming 0.1 is still optimal (report 09 §Re-Tuning) |
| `experiments/11-priority-queue/`  | `binary-v4-pq` (`37676d5`) | FIFO vs shared largest-first queue; logs, nsys, A/B (negative result) |
| `experiments/12-gpu-affinity/`    | `binary-v5-affinity` (`fc33e29`) | hybrid A/B (−3.6 %), nsys, CPU-only 3-way ordering study |
| `experiments/13-zoom-points/`     | `binary-v5-affinity` (`fc33e29`) | four-regime characterization (outside/inside/Misiurewicz/seahorse), per-point logs |
| `experiments/16-ncu-divergence/`  | `binary-v5-affinity` (`fc33e29`) vs ncu 2026.2 | Nsight **Compute** kernel measurement — warp efficiency / FP64 / roofline across four content regimes; `capture.sh` (sudo), `specs/`, `analysis/summary.csv` (report 16) |
| `experiments/17-ncu-zoom-points/` | `binary-v5-affinity` (`fc33e29`) vs ncu 2026.2 | report 16's method re-run on the four report-13 zoom geometries (report 17) |
| `experiments/18-maxiter-split/`   | `examine/maxiter-split` (`d572c78`) | leaf-count equivalence of the parameter-free split rule vs `diffT=0.1` (report 18) |
| `experiments/20-igpu-opencl/`     | `feat/igpu-opencl` atop v5 | iGPU third-executor sweep + CUDA-vs-OpenCL output verification (report 20) |
| `experiments/21-periodicity-check/` | `binary-v6-periodicity` (`fa6d8ac`) vs `main` | CPU Brent periodicity A/B (−78.6 % CPU12, −46.3 % hybrid) + `verify_identity.sh` 100/100 (report 21) |
| `experiments/22-igpu-atop-v6/`    | `feat/igpu-opencl` atop v6 | iGPU re-price after periodicity (a wash) + blocking-sync and oversubscription probes (report 22) |
| `experiments/23-gpu-periodicity/` | `binary-v7-gpu-periodicity` (`2abc909`) vs `main` | GPU Brent periodicity A/B (kernel 2.34×, hybrid 23.72 s) + identity check; AC-gated `ab.sh` (report 23) |
| `experiments/24-ncu-gpu-periodicity/` | `binary-v7` (`2abc909`) vs ncu 2026.2 | ncu re-baseline after the GPU check — the “degraded” warp metrics are eliminated waste (report 24) |
| `experiments/25-frame-distribution/` | `binary-frame-dist`+ (branch lineage) | `verify_dist.sh` identity checks, laptop+ivy and laptop+yeco benchmark drivers and CSVs (static, weighted, dynamic-stealing, pre/post tar batching) — reports 25/26/27 |
| `experiments/28-dlt-analysis/`    | analysis (external DLTlib + GLPK) | DLT solver driver, `run.sh`, `results.txt`, `sweep.csv` — closed-form retrodiction of the distribution campaign (report 28) |

`*.nsys-rep`, `*.ncu-rep`, `*.sqlite`, `*.png`, and plain `*.log` files are gitignored;
per-rep `*.stderr`/`*.stdout` files are tracked so the reports' numbers can be
re-derived without rerunning. Each report file (`reports/NN-name.tex`) names the
experiment directory it draws from. (There are no reports 10/19; experiment
10's re-tuning sweep is written up as a section of report 09. There is no
`experiments/03` — report 03 is pure code analysis — and none for reports
14/15, which are analysis/guide documents. Reports 26 and 27 draw their data
from `experiments/25-frame-distribution/`, which hosts the whole
distribution campaign.)
