#include <stdio.h>
#include <cuda.h>
#include <kat/on_device/c_standard_library/printf.cuh>

__global__ void hello_world() {
  char buffer [100];
  kat::sprintf(buffer, ", and the half of that is %d.", 60/2/2 );
  //puts(buffer);
}

int main() {
  hello_world<<<1, 1>>>();
  cudaDeviceSynchronize();

  return 0;
}
