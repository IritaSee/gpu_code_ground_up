#include <stdio.h>
#include <cuda_runtime.h>
#include <cufile.h>

#define CUDA_CHECK(status) \
    do { \
        cudaError_t err = status; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

#define CUFILE_CHECK(status) \
    do { \
        cufile_error_t err = status; \
        if (err != CUFILE_SUCCESS) { \
            fprintf(stderr, "cuFile error: %s at line %d\n", cufile_error_string(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

int main() {
    // Allocate and initialize data on the GPU
    const int numElements = 1024;
    const int bufferSize = numElements * sizeof(float);
    float* d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, bufferSize));
    CUDA_CHECK(cudaMemset(d_data, 0, bufferSize));

    // Open the output file using cuFile
    const char* filePath = "output_file.bin";
    cufile_handle_t handle;
    CUFILE_CHECK(cuFileDriverOpen(&handle, CUFILE_DRIVER_GDS));

    // Write data from GPU to the file using cuFile
    const off_t offset = 0;
    CUFILE_CHECK(cuFileWrite(handle, d_data, bufferSize, offset, 0));

    // Close the file handle
    CUFILE_CHECK(cuFileDriverClose(handle));

    // Free allocated memory on the GPU
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}