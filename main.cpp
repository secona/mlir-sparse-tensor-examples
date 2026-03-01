#include <cstdint>
#include <mlir/Dialect/SparseTensor/IR/Enums.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <mlir/ExecutionEngine/SparseTensorRuntime.h>
#include <mlir/ExecutionEngine/SparseTensor/File.h>
#include <cstdlib>

struct MemRefDescriptor2D {
  float* allocated;
  float* aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

extern "C" {
    void _mlir_ciface_tensor_add(MemRefDescriptor2D *out, void *a, void *b);
}

void* create_empty_csr_tensor() {
    uint64_t sizes_data[] = {32, 32};
    StridedMemRefType<uint64_t, 1> sizes_memref = {sizes_data, sizes_data, 0, {2}, {1}};

    LevelType lvl_types_data[] = {LevelType(LevelFormat::Dense), LevelType(LevelFormat::Compressed)};
    StridedMemRefType<LevelType, 1> lvl_types_memref = {lvl_types_data, lvl_types_data, 0, {2}, {1}};

    uint64_t mapping_data[] = {0, 1};
    StridedMemRefType<uint64_t, 1> mapping_memref = {mapping_data, mapping_data, 0, {2}, {1}};

    uint64_t dimRank = 2;
    uint64_t dimShape[2] = {32, 32};
    PrimaryType valTp = PrimaryType::kF32;
    SparseTensorReader *reader = SparseTensorReader::create("ibm32.mtx", dimRank, dimShape, valTp);

    void *tensor = _mlir_ciface_newSparseTensor(
        &sizes_memref, 
        &sizes_memref, 
        &lvl_types_memref, 
        &mapping_memref, 
        &mapping_memref, 
        mlir::sparse_tensor::OverheadType::kIndex, 
        mlir::sparse_tensor::OverheadType::kIndex, 
        mlir::sparse_tensor::PrimaryType::kF32, 
        mlir::sparse_tensor::Action::kFromReader, 
        reader
    );

    delSparseTensorReader(reader);

    return tensor;
}

int main() {
    void *sparse_tensor_a = create_empty_csr_tensor();
    void *sparse_tensor_b = create_empty_csr_tensor();
    MemRefDescriptor2D out;

    _mlir_ciface_tensor_add(&out, sparse_tensor_a, sparse_tensor_b);

    delSparseTensor(sparse_tensor_a);
    delSparseTensor(sparse_tensor_b);

    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            int idx = i * 32 + j;
            if (out.aligned[idx] == 0)
                std::cout << "  ";
            else 
                std::cout << out.aligned[idx] << " ";
        }
        std::cout << "\n";
    }

    std::free(out.allocated);
}
