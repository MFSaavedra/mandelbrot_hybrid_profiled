# GPU=0: CPU-only build for nodes with no usable CUDA stack (e.g. Kepler
# sm_30, dropped after CUDA 10.2 — the GT 750M node).  Skips nvcc entirely:
# kernel_stub.cpp satisfies the linker (the GPU path is only called when
# gpuEnable=1; the stub fails loudly if it is), NVTX becomes a no-op
# (-DNO_NVTX), the system g++ compiles and links (the g++-15 pin exists only
# for nvcc compatibility).  Run `make clean` when switching GPU=0 <-> GPU=1 —
# the object files are not flavour-tagged.
GPU ?= 1

# CUDA install root: /opt/cuda on Arch, /usr/local/cuda (version symlink) on
# Ubuntu.  Overridable: make CUDAINST=/usr/local/cuda-13.3
CUDAINST ?= $(firstword $(wildcard /opt/cuda /usr/local/cuda))

ifeq ($(GPU),0)
HOSTCC   ?= g++
CC        = $(HOSTCC)
else
# nvcc needs a host gcc it supports: Arch's system gcc 16 is too new for nvcc
# 13.2, so pin g++-15 where it exists; elsewhere the system g++ is fine (e.g.
# Ubuntu 24.04's gcc 13 under nvcc 13.3).  -ccbin keeps both halves of the
# program on the same ABI.  Invoke the toolkit's own nvcc, not whatever is
# first in PATH (Ubuntu's /usr/bin/nvcc is a stale apt toolkit).
HOSTCC   ?= $(shell command -v g++-15 >/dev/null 2>&1 && echo g++-15 || echo g++)
NVCC      = $(CUDAINST)/bin/nvcc -ccbin $(HOSTCC)
CC        = $(HOSTCC)
endif

# ---------------------------------------------------------------------------
# Qt5: use pkg-config so the paths are correct on any distro (Arch, Ubuntu…)
# On Arch:  headers → /usr/include/qt/   libs → /usr/lib
# On Ubuntu: headers → /usr/include/qt5/ libs → /usr/lib/x86_64-linux-gnu
# ---------------------------------------------------------------------------
QT_CFLAGS = $(shell pkg-config --cflags Qt5Core Qt5Gui)
QT_LIBS   = $(shell pkg-config --libs   Qt5Core Qt5Gui)

# ---------------------------------------------------------------------------
# CUDA architecture
#   -arch=native   → auto-detects the GPU in this machine (CUDA 11.6+)
#   Fallback if your nvcc is older: use -arch=sm_61 (Pascal, GTX 10xx)
#   compute_20/sm_21 (Fermi) was removed in CUDA 12 — do NOT use it.
# ---------------------------------------------------------------------------
CUDA_ARCH = -arch=native

CUDA_LINK_FLAGS    = -rdc=true $(CUDA_ARCH)
CUDA_COMPILE_FLAGS = -O2 --device-c $(CUDA_ARCH)

CC_COMPILE_FLAGS = -O2 -I$(CUDAINST)/include $(QT_CFLAGS) -fPIC
# NVTX3 (CUDA 11+) is header-only: it dlopen()s the profiler's runtime library
# when Nsight Systems is active, so no -lnvToolsExt is needed.  -ldl provides
# the dlopen symbol that the NVTX3 headers use internally.
CC_LINK_FLAGS    = -lm -lstdc++ $(QT_LIBS) -lpthread -ldl

ifeq ($(GPU),0)
CC_COMPILE_FLAGS += -DNO_NVTX
KERNEL_O  = kernel_stub.o
LINK      = $(CC)
else
KERNEL_O  = kernel.o
LINK      = $(NVCC) $(CUDA_LINK_FLAGS)
endif

HEADERS = kernel.h mandelframe.h mandelregion.h workqueue.h

# ---------------------------------------------------------------------------
all: mandelHybrid

mandelHybrid: main.o mandelframe.o mandelregion.o workqueue.o $(KERNEL_O)
	$(LINK) $^ $(CC_LINK_FLAGS) -o $@

main.o: main.cpp $(HEADERS)
	$(CC) $(CC_COMPILE_FLAGS) -c main.cpp

mandelframe.o: mandelframe.cpp $(HEADERS)
	$(CC) $(CC_COMPILE_FLAGS) -c mandelframe.cpp

mandelregion.o: mandelregion.cpp $(HEADERS)
	$(CC) $(CC_COMPILE_FLAGS) -c mandelregion.cpp

workqueue.o: workqueue.cpp $(HEADERS)
	$(CC) $(CC_COMPILE_FLAGS) -c workqueue.cpp

kernel.o: kernel.cu kernel.h
	$(NVCC) $(CUDA_COMPILE_FLAGS) -c kernel.cu -o $@

kernel_stub.o: kernel_stub.cpp kernel.h
	$(CC) $(CC_COMPILE_FLAGS) -c kernel_stub.cpp

clean:
	rm -f *.o mandelHybrid
