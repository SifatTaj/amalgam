

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

    if (idx < n_channels * batch_size) {
        int aug_input_offset = idx * aug_input_size;
        int filtered_input_offset = idx * filtered_input_size;
        int channel_idx = idx % n_channels;
        int aug_indices_offset = channel_idx * aug_indices_size;

        int filtered_idx = 0;
        int aug_count = 0;
        
        for (int i = 0; i < aug_input_size; ++i) {
            bool is_augmented = false;
            int aug_input_idx = aug_input_offset + i;

            for (int j = 0; j < aug_indices_size; ++j) {
                int aug_index = aug_indices[aug_indices_offset + j];

                if (i == aug_index) {
                    is_augmented = true;
                    aug_count++;
                    break;
                }
            }

            if (!is_augmented) {
                if (filtered_idx < filtered_input_size) {
                    filtered_input[filtered_input_offset + filtered_idx] = aug_input[aug_input_idx];
                    filtered_idx++;
                }
            }
        }
    }
}