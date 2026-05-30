# Instrumentation reference

The `mandelHybrid` binary is heavily instrumented for profiling. This document
describes each mechanism, what it measures, and how to consume its output. All
of it operates at near-zero overhead when no profiler is attached.

For results derived from this instrumentation see the reports under
[`reports/`](reports/) and the raw data under
[`experiments/`](experiments/).

---

## Running the binary

```bash
./mandelHybrid spec.in <numThreads> <gpuEnable> [diffThreshold] [pixelThreshold] [quiet] [save]
```

| arg | default | meaning |
|---|---|---|
| `numThreads` | core count | total CPU+GPU worker threads |
| `gpuEnable` | 1 | 1 enables GPU; 0 is CPU-only |
| `diffThreshold` | 0.5 (legacy) | uniform when `(maxCornerIter − minCornerIter) < diffT × maxCornerIter`. **Recommended new default: 0.1** (see report 06) |
| `pixelThreshold` | 32768 | regions smaller than this are computed without further splitting |
| `quiet` | 0 | 1 suppresses the per-region `[CPU region NNN]` / `[GPU region NNN]` lines; per-thread and CUDAmemCleanup summaries still emit |
| `save` | 1 | 0 skips `img->save()` so pure compute can be timed |

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
  the full decision cycle for one region: corner evaluation plus either a
  pixel compute or a four-way split back onto the work queue. Lets you
  visually count how many `examine()` calls each thread handles and compare
  depth/duration distributions between CPU and GPU workers.
- **`CPU region compute`** — `mandelregion.cpp`, `MandelRegion::compute`
  (CPU branch). Wraps the pixel iteration loop for CPU-computed regions.
- **`GPU region compute`** — `kernel.cu`, `hostFE`. Wraps the full GPU path:
  kernel launch + device-to-host memcpy + result copy into the QImage buffer.
  Appears on thread 0's lane alongside the automatically recorded CUDA API
  calls.

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
[GPU region   42] size  960x 540  kernel 1.234 ms  D->H 0.456 ms
```

`CUDAmemCleanup()` prints an aggregate summary at shutdown:

```
[GPU profiling summary]
  Regions computed on GPU : 3743
  Total kernel time       : 46.50 s   (avg 12.44 ms/region)
  Total D->H memcpy time  :  0.090 s  (avg 24.0 µs/region)
```

### 3. CPU thread timing  (`mandelregion.cpp`)

`clock_gettime(CLOCK_MONOTONIC)` brackets the pixel iteration loop for every
CPU-computed region. `thread_local` counters accumulate region count and total
compute time per thread with no synchronization overhead.

Each CPU region prints a line to stderr (unless `quiet=1`):

```
[CPU region    7] size  480x 270  compute 3.210 ms
```

When a thread's work queue drains, `MandelRegion::printCPUSummary()` prints:

```
[CPU profiling summary - this thread]
  Regions computed on CPU : 7
  Total compute time      : 22.470 ms  (avg 3.210 ms/region)
```

The GPU thread (thread 0) prints nothing from `printCPUSummary()` because its
regions are counted inside `hostFE()` and summarised in `CUDAmemCleanup()`.

### 4. End-to-end wall timer  (`main.cpp`)

`clock_gettime(CLOCK_MONOTONIC)` brackets the compute phase from
`CUDAmemSetup` (when GPU is enabled) through every worker thread's `wait()`
return. Prints exactly one machine-parseable line at shutdown:

```
[total_elapsed_s] 53.241145
```

`scripts/sweep_fig1115.sh` greps for this line to populate its CSV. Image
saves are included inside the bracket because they happen on whichever CPU
worker decrements `remainingRegions` to 0.

Also printed at startup is a configuration banner so log files are
self-describing:

```
[config] numThr=12 gpuEnable=1 diffT=0.500 pixT=32768 res=1920x1080 frames=100 quiet=0 save=1
```

`CLOCK_MONOTONIC` is used (rather than `CLOCK_REALTIME`) so NTP/DST jumps
cannot corrupt the measurement.

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

What to look for in the timeline:

- NVTX rows show `examine region`, `CPU region compute`, and
  `GPU region compute` bars side by side with CUDA API calls.
- Compare how much wall time each `CalcThr` lane spends on GPU vs CPU work.
- GPU idle time at the end of the run indicates CPU threads are still
  finishing the tail; quantifies the imbalance.
- The CUDA row shows kernel + memcpy back-to-back; a gap between them would
  indicate CPU-side stalls feeding the queue.
- Note that `cudaMemcpy` reports its full wait time including the implicit
  stream synchronization — most of its reported duration is kernel-wait, not
  actual transfer. See report 01 for the breakdown.

### Nsight Compute (kernel deep-dive)

```bash
ncu --set full -o kernel_report ./mandelHybrid spec.in 1 1
ncu-ui kernel_report.ncu-rep
```

Limit to 1 thread and 1 frame on the first pass: `ncu` replays each kernel
launch multiple times to collect all hardware counter sets, so a full run
with many regions is very slow.

What to look for:

- Occupancy — the 16×16 thread block gives 256 threads per launch; check
  whether the warp scheduler is fully utilised.
- Memory throughput — `mandelKernel` is compute-bound (no global reads after
  the kernel starts), but the D → H memcpy bandwidth is visible here.
- Warp stall reasons — long-divergence regions near the Mandelbrot boundary
  cause branch divergence; `ncu` flags this.

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

`*.nsys-rep`, `*.sqlite`, `*.png`, and plain `*.log` files are gitignored;
per-rep `*.log.stderr` and `*.log.stdout` files are tracked so the reports'
numbers can be re-derived without rerunning the sweeps. Each report file
itself (`reports/NN-name.tex`) names the experiment directory it draws from.
