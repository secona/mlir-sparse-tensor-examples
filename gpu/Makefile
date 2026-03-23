CC    := clang
CXX   := clang++
OPT   := mlir-opt
TRANS := mlir-translate

CUDA_PATH  ?= /usr/local/cuda-13.0
LLVM_BUILD ?= /home/vito/llvm-project/build

INCLUDES := -I$(CUDA_PATH)/targets/x86_64-linux/include
LDFLAGS  := -L$(LLVM_BUILD)/lib -Wl,-rpath,$(LLVM_BUILD)/lib
LIBS     := -lmlir_cuda_runtime -lmlir_runner_utils -lcudart

CXXFLAGS := -O3
GPU_OPTS := -gpu-lower-to-nvvm-pipeline='cubin-triple=nvptx64-nvidia-cuda cubin-chip=sm_75 cubin-features=+ptx63'

OUTPUT := program

.PHONY: all clean

all: $(OUTPUT)

$(OUTPUT): main.cpp kernel.o
	$(CXX) $^ $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@

lowered.mlir: matrix_add.mlir
	$(OPT) $< $(GPU_OPTS) > $@

kernel.ll: lowered.mlir
	$(TRANS) -mlir-to-llvmir $< > $@

%.o: %.ll
	$(CC) -Wall -c $< -o $@ -O3

clean:
	rm -f kernel.ll kernel.o lowered.mlir $(OUTPUT)
