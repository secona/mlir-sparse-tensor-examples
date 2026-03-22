#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>

template <typename T>
struct MemRef2D {
    T* allocated_ptr;
    T* aligned_ptr;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

extern "C" void _mlir_ciface_vector_add_gpu(MemRef2D<float>* A, MemRef2D<float>* B, MemRef2D<float>* C, intptr_t d);

int main() {
    const int N = 1024;
    const int size = N * N;

    float *shared_A, *shared_B, *shared_C;
    cudaMallocManaged(&shared_A, size * sizeof(float));
    cudaMallocManaged(&shared_B, size * sizeof(float));
    cudaMallocManaged(&shared_C, size * sizeof(float));

    for(int i = 0; i < size; i++) {
        shared_A[i] = 1.0f;
        shared_B[i] = 2.0f;
        shared_C[i] = 0.0f;
    }

    MemRef2D<float> mem_A = {shared_A, shared_A, 0, {N, N}, {N, 1}};
    MemRef2D<float> mem_B = {shared_B, shared_B, 0, {N, N}, {N, 1}};
    MemRef2D<float> mem_C = {shared_C, shared_C, 0, {N, N}, {N, 1}};

    std::cout << "Executing MLIR GPU Kernel..." << std::endl;
    _mlir_ciface_vector_add_gpu(&mem_A, &mem_B, &mem_C, size);

    std::cout << "Result at [0,0] (Expected 3.0): " << shared_C[0] << std::endl;

    cudaFree(shared_A);
    cudaFree(shared_B);
    cudaFree(shared_C);

    return 0;
}
