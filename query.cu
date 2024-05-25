#include <stdio.h>

int main() {

  int nDevices;
  cudaGetDeviceCount(&nDevices);

  printf("Number of devices: %d\n", nDevices);

  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device name: %s\n", prop.name);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("Max Blocks per Multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per MP: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("\n");
  }
}