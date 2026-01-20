/**
 * @brief A CUDA kernel to filter out the augmented indices. Each thread filters one channel. 
 * @param aug_input Augmented input of size batch_size * n_channels * sample_height * sample_width.
 * @param filtered_input Filtered input without the augmented indices.
 * @param aug_indices List of augmented indices.
 * @param aug_indices_size Size of augmented indices for each channel.
 * @param batch_size Size of the batch.
 * @param aug_input_size Size of the augmented sample.
 * @param filtered_input_size Size of the filtered (deanon) sample.
 * @param n_channels Number of channels.
 */

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check idx out of bound
    if (idx < n_channels * batch_size) {

        // Calculate the offset for each thread
        int aug_input_offset = idx * aug_input_size;
        int filtered_input_offset = idx * filtered_input_size;
        int channel_idx = idx % n_channels;
        int aug_indices_offset = channel_idx * aug_indices_size;

        int filtered_idx = 0;
        int aug_count = 0;
        
        // Iterate through augmented sample
        for (int i = 0; i < aug_input_size; ++i) {
            bool is_augmented = false;
            int aug_input_idx = aug_input_offset + i;

            // Iterate through the augmented indices to check if index i is augmented or not
            // TODO: Can be further optimized with binary search
            // TODO: Try using sets for contast lookups
            for (int j = 0; j < aug_indices_size; ++j) {
                int aug_index = aug_indices[aug_indices_offset + j];

                if (i == aug_index) {
                    is_augmented = true;
                    aug_count++;
                    break;
                }
            }

            // Copy data if not augmtented
            if (!is_augmented) {
                if (filtered_idx < filtered_input_size) {
                    filtered_input[filtered_input_offset + filtered_idx] = aug_input[aug_input_idx];
                    filtered_idx++;
                }
            }
        }
    }
}