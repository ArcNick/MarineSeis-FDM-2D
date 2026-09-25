NVCC = nvcc

# Target GPU architecture.  The current machine has an RTX 5060 (sm_120);
# override on other machines, e.g.  make ARCH=sm_90   or   make ARCH=native
# ("native" requires the GPU to be visible at build time).
ARCH ?= sm_120

CXXFLAGS = -O3 -std=c++17
# -rdc=true is required for the `extern __constant__` grid constants to be
# shared across translation units (otherwise nvcc makes each one a per-TU
# static definition and cudaMemcpyToSymbol fails).

# ARCHFLAGS = -arch=$(ARCH) -lineinfo -rdc=true
ARCHFLAGS = -lineinfo -rdc=true
INCLUDES = -Iinclude
LDFLAGS = -Llib -lcjson -lm

TARGET = bin/main_debug
SOURCES = src/*.cu

HEADERS = $(wildcard include/*.h) $(wildcard include/*.cuh) $(wildcard include/*.hpp)

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(CXXFLAGS) $(ARCHFLAGS) $(INCLUDES) $(SOURCES) $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean
