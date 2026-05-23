# nvcc 13.2 does not support gcc 16 yet; pin to g++-15 for the host compiler
# and for nvcc's -ccbin so both halves of the program share an ABI.
HOSTCC    = g++-15
NVCC      = nvcc -ccbin $(HOSTCC)
CC        = $(HOSTCC)
CUDAINST  = /opt/cuda

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

HEADERS = kernel.h mandelframe.h mandelregion.h workqueue.h

# ---------------------------------------------------------------------------
all: mandelHybrid

mandelHybrid: main.o mandelframe.o mandelregion.o workqueue.o kernel.o
	$(NVCC) $(CUDA_LINK_FLAGS) $^ $(CC_LINK_FLAGS) -o $@

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

clean:
	rm -f *.o mandelHybrid
