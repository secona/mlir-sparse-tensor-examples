#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mlir/Dialect/SparseTensor/IR/Enums.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <mlir/ExecutionEngine/SparseTensor/File.h>
#include <mlir/ExecutionEngine/SparseTensorRuntime.h>
#include <string>

typedef StridedMemRefType<float, 2> MemRef2D;

extern "C" {
void _mlir_ciface_tensor_add(MemRef2D *out, void *a);
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

  uint64_t nse = getSparseTensorReaderNSE(reader);
  std::cout << "non-zeros: " << nse << std::endl;

  delSparseTensorReader(reader);

  return tensor;
}

int main() {
  void *sparse_tensor_a = create_csr_tensor("ibm32.mtx");
  assert(sparse_tensor_a);

  MemRef2D *out = (MemRef2D *)malloc(sizeof(MemRef2D));

  _mlir_ciface_tensor_add(out, sparse_tensor_a);

  std::cout << "offset: " << out->offset << "\n";
  std::cout << "sizes: " << out->sizes[0] << " " << out->sizes[1] << "\n";
  std::cout << "strides: " << out->strides[0] << " " << out->strides[1] << "\n";

  float *data = out->data;

  for (int i = 0; i < out->sizes[0]; i++) {
    for (int j = 0; j < out->sizes[1]; j++) {
      int idx = out->offset + i * out->strides[0] + j * out->strides[1];
      float val = data[idx];

      if (val) std::cout << val << " ";
      else std::cout << ". ";
    }
    std::cout << "\n";
  }

  std::free(out->basePtr);
  delSparseTensor(sparse_tensor_a);
}
