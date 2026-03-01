CXX     = clang++
CC      = clang
LLC     = llc
MLIROPT = mlir-opt
MLIRTRN = mlir-translate

CXXFLAGS = -std=c++17 -Wall -g -fsanitize=address
LDFLAGS  = -lmlir_c_runner_utils -lmlir_runner_utils

TARGET = tensor_runner

OBJS = main.o tensor_add.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -o $(TARGET)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

tensor_add_opt.mlir: tensor_add.mlir
	$(MLIROPT) tensor_add.mlir -pass-pipeline='builtin.module(sparsifier)' > tensor_add_opt.mlir

tensor_add.ll: tensor_add_opt.mlir
	$(MLIRTRN) --mlir-to-llvmir tensor_add_opt.mlir -o tensor_add.ll

tensor_add.o: tensor_add.ll
	$(LLC) -filetype=obj tensor_add.ll -o tensor_add.o

clean:
	rm -f $(OBJS) $(TARGET) tensor_add_opt.mlir tensor_add.ll
