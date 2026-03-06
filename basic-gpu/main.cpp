#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mlir/Dialect/SparseTensor/IR/Enums.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <mlir/ExecutionEngine/SparseTensor/File.h>
#include <mlir/ExecutionEngine/SparseTensorRuntime.h>
#include <string>

// See https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission
struct MemRefDescriptor2D {
  float *allocated;
  float *aligned;
  intptr_t offset;
  intptr_t sizes[2];
  intptr_t strides[2];
};

extern "C" {
void _mlir_ciface_tensor_add(MemRefDescriptor2D *out, void *a);
}

void *create_csr_tensor(std::string filename) {
  uint64_t sizes_data[] = {32, 32};
  StridedMemRefType<uint64_t, 1> sizes_memref = {
      sizes_data, sizes_data, 0, {2}, {1}};

  LevelType lvl_types_data[] = {LevelType(LevelFormat::Dense),
                                LevelType(LevelFormat::Compressed)};
  StridedMemRefType<LevelType, 1> lvl_types_memref = {
      lvl_types_data, lvl_types_data, 0, {2}, {1}};

  uint64_t mapping_data[] = {0, 1};
  StridedMemRefType<uint64_t, 1> mapping_memref = {
      mapping_data, mapping_data, 0, {2}, {1}};

  uint64_t dimRank = 2;
  uint64_t dimShape[2] = {32, 32};
  PrimaryType valTp = PrimaryType::kF32;
  SparseTensorReader *reader =
      SparseTensorReader::create(filename.c_str(), dimRank, dimShape, valTp);

  void *tensor = _mlir_ciface_newSparseTensor(
      &sizes_memref, &sizes_memref, &lvl_types_memref, &mapping_memref,
      &mapping_memref, mlir::sparse_tensor::OverheadType::kIndex,
      mlir::sparse_tensor::OverheadType::kIndex, valTp,
      mlir::sparse_tensor::Action::kFromReader, reader);

  delSparseTensorReader(reader);

  return tensor;
}

int main() {
  void *sparse_tensor_a = create_csr_tensor("ibm32.mtx");
  MemRefDescriptor2D *out = (MemRefDescriptor2D *)malloc(sizeof(MemRefDescriptor2D));

  _mlir_ciface_tensor_add(out, sparse_tensor_a);

  delSparseTensor(sparse_tensor_a);

  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 32; j++) {
      int idx = i * 32 + j;
      if (out->aligned[idx] == 0)
        std::cout << "  ";
      else
        std::cout << out->aligned[idx] << " ";
    }
    std::cout << "\n";
  }

  std::free(out->allocated);
}
