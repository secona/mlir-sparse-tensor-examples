LLVM_ROOT    := /home/guest/llvm-project
LLVM_BUILD   := $(LLVM_ROOT)/build
MLIR_INC     := $(LLVM_ROOT)/mlir/include
MLIR_LIB     := $(LLVM_BUILD)/lib

CXX      := clang++
LLC      := $(LLVM_BUILD)/bin/llc
MLIROPT  := $(LLVM_BUILD)/bin/mlir-opt
MLIRTRN  := $(LLVM_BUILD)/bin/mlir-translate

CXXFLAGS := -std=c++17 -Wall -g -I$(MLIR_INC)
LDFLAGS  := -L$(MLIR_LIB) -Wl,-rpath,$(MLIR_LIB) \
            -lmlir_c_runner_utils -lmlir_runner_utils -lmlir_cuda_runtime

GPU_SPARSIFIER := "enable-gpu-libgen gpu-numthreads=32 gpu-triple=nvptx64-nvidia-cuda \
                   gpu-chip=sm_75 gpu-features=+ptx80 enable-runtime-library=true \
                   parallelization-strategy=dense-outer-loop gpu-format=bin"

TARGET := tensor_runner
OBJS   := main.o tensor_add.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

tensor_add_opt.mlir: tensor_add.mlir
	$(MLIROPT) $< --sparsifier=$(GPU_SPARSIFIER) -o $@

tensor_add.ll: tensor_add_opt.mlir
	$(MLIRTRN) --mlir-to-llvmir $< -o $@

tensor_add.o: tensor_add.ll
	$(LLC) -relocation-model=pic -filetype=obj $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) tensor_add_opt.mlir tensor_add.ll
