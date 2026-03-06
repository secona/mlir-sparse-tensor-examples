CXX     = clang++
CC      = clang
LLC     = llc
MLIROPT = mlir-opt
MLIRTRN = mlir-translate

LLVM_DIR ?= /home/vito/llvm-project/build
MLIR_DIR ?= $(LLVM_DIR)

MLIR_LIB_DIR ?= $(MLIR_DIR)/lib
MLIR_INC_DIR ?= /home/vito/llvm-project/mlir/include

CXXFLAGS = -std=c++17 -Wall -g -I$(MLIR_INC_DIR)
LDFLAGS  = -L$(MLIR_LIB_DIR) -Wl,-rpath,$(MLIR_LIB_DIR) -lmlir_c_runner_utils -lmlir_runner_utils -lmlir_cuda_runtime

GPU_SPARSIFIER = enable-gpu-libgen gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_75 gpu-features=+ptx80 enable-runtime-library=true parallelization-strategy=dense-outer-loop gpu-format=bin

TARGET = tensor_runner

OBJS = main.o tensor_add.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -o $(TARGET)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

tensor_add_opt.mlir: tensor_add.mlir
	$(MLIROPT) tensor_add.mlir --sparsifier="$(GPU_SPARSIFIER)" > tensor_add_opt.mlir

tensor_add.ll: tensor_add_opt.mlir
	$(MLIRTRN) --mlir-to-llvmir tensor_add_opt.mlir -o tensor_add.ll

tensor_add.o: tensor_add.ll
	$(LLC) -relocation-model=pic -filetype=obj tensor_add.ll -o tensor_add.o

clean:
	rm -f $(OBJS) $(TARGET) tensor_add_opt.mlir tensor_add.ll
