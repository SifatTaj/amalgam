#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include <cuda_runtime.h>

// Error checking macros for CUDA and cuBLAS
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

__global__ void filter_aug_input_kernel(
    const float* aug_input,
    float* filtered_input,
    const int* aug_indices,
    const int aug_indices_size,
    const int batch_size,
    const int aug_input_size,
    const int filtered_input_size,
    const int n_channels
) {
    int channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (channel_idx < n_channels && batch_idx < batch_size) {

        int aug_input_offset = batch_idx * n_channels * aug_input_size + channel_idx * aug_input_size;
        int filtered_input_offset = batch_idx * n_channels * filtered_input_size + channel_idx * filtered_input_size;
        int aug_indices_offset = channel_idx * aug_indices_size;

        int filtered_idx = 0;
        
        for (int idx = 0; idx < aug_input_size; ++idx) {
            bool is_augmented = false;
            int aug_input_idx = aug_input_offset + idx;
            for (int j = 0; j < aug_indices_size; ++j) {
                int aug_index = aug_indices[aug_indices_offset + j];
                if (aug_input_idx == aug_index) {
                    is_augmented = true;
                    break;
                }
            }
            if (!is_augmented) {
                filtered_input[filtered_input_offset + filtered_idx] = aug_input[aug_input_idx];
                filtered_idx++;
            }
        }
    }
}

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
        size_t aug_input_size_t = batch_size * n_channels * aug_input_size * sizeof(float);
        size_t filtered_input_size_t = batch_size * n_channels * filtered_input_size * sizeof(float);
        size_t aug_indices_size_t = n_channels * aug_indices_size * sizeof(int);

        float *d_aug_input;
        float *d_filtered_input;
        int *d_aug_indices;

        CHECK_CUDA(cudaMalloc(&d_aug_input, aug_input_size_t));
        CHECK_CUDA(cudaMalloc(&d_filtered_input, filtered_input_size_t));
        CHECK_CUDA(cudaMalloc(&d_aug_indices, aug_indices_size_t));

        CHECK_CUDA(cudaMemcpy(d_aug_input, aug_input, aug_input_size_t, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_filtered_input, filtered_input, filtered_input_size_t, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_aug_indices, aug_indices, aug_indices_size_t, cudaMemcpyHostToDevice));

        dim3 threads_per_block(3, 32);
        dim3 blocks_per_grid(
            (n_channels + threads_per_block.x - 1) / threads_per_block.x,
            (batch_size + threads_per_block.y - 1) / threads_per_block.y
        );

        filter_aug_input_kernel<<<blocks_per_grid, threads_per_block>>>(
            d_aug_input,
            d_filtered_input,
            d_aug_indices,
            aug_indices_size,
            batch_size,
            aug_input_size,
            filtered_input_size,
            n_channels
        );

        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(filtered_input, d_filtered_input, filtered_input_size * sizeof(float), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_aug_input));
        CHECK_CUDA(cudaFree(d_filtered_input));
        CHECK_CUDA(cudaFree(d_aug_indices));
    }
}