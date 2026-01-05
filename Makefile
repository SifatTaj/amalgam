.PHONY: filter_input clean

CUDA_SRC_DIR = core/
CUDA_OUT_DIR = core/lib

all: filter_input

filter_input:
	nvcc -shared -o  $(CUDA_OUT_DIR)/filter_input.so --compiler-options '-fPIC' $(CUDA_SRC_DIR)/fliter_input.cu

clean:
	rm -r $(CUDA_OUT_DIR)/filter_input.so