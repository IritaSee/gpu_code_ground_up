#include <stdio.h>
#include <cuda.h>
#include <kat/

__global__ void hello_world() {
  kat::snprintf("Hello, world from CUDA!\n");
}

int main() {
  hello_world<<<1, 1>>>();
  cudaDeviceSynchronize();

  return 0;
}