#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include <cuda_runtime.h>
#include "filter_kernel.cuh"

// Error checking macros for CUDA and cuBLAS
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

const int THREADS_PER_BLOCK = 1024;

extern "C" {
    void filter_aug_input(
        float* aug_input,
        float* filtered_input,
        int* aug_indices,
        int aug_indices_size,
        int batch_size,
        int aug_input_size,
        int filtered_input_size,
        int n_channels
    ) {
        int threads_per_block = THREADS_PER_BLOCK;
        int blocks_per_grid = ((n_channels * batch_size) + threads_per_block - 1) / threads_per_block;

        filter_aug_input_kernel<<<blocks_per_grid, threads_per_block>>>(
            aug_input,
            filtered_input,
            aug_indices,
            aug_indices_size,
            batch_size,
            aug_input_size,
            filtered_input_size,
            n_channels
        );

        CHECK_CUDA(cudaDeviceSynchronize());
    }
}