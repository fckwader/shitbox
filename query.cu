#include <stdio.h>

int main() {

  int nDevices;
  cudaGetDeviceCount(&nDevices);

  printf("Number of devices: %d\n", nDevices);

  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
  }
}