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

/**
 * @brief Initize the CUDA kernel to filter the augmented input. 
 * @param aug_input Augmented input of size batch_size * n_channels * sample_height * sample_width.
 * @param filtered_input Filtered input without the augmented indices.
 * @param aug_indices List of augmented indices.
 * @param aug_indices_size Size of augmented indices for each channel.
 * @param batch_size Size of the batch.
 * @param aug_input_size Size of the augmented sample.
 * @param filtered_input_size Size of the filtered (deanon) sample.
 * @param n_channels Number of channels.
 */

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
        // Configure kernel
        int threads_per_block = THREADS_PER_BLOCK;
        int blocks_per_grid = ((n_channels * batch_size) + threads_per_block - 1) / threads_per_block;

        // Launch kernel
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