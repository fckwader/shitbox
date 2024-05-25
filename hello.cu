#include <stdio.h>




__global__
void add(int n, float *x, float *y, float *z)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride){
        z[i] = x[i] + y[i];
    }
}

int main()
{
    int n = 1000000;
    float *x, *y, *z;
    cudaMallocManaged(&x, n*sizeof(float));
    cudaMallocManaged(&y, n*sizeof(float));
    cudaMallocManaged(&z, n*sizeof(float));

    //init
    printf("Init...\n");
    for(int i = 0; i < n; i++){
        x[i] = i % 8;
        y[i] = i % 25;
        z[i] = 0;
    }
    printf("Init complete\n");

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    printf("Calc...\n");
    add<<<256, 256>>>(n, x, y, z);
    cudaDeviceSynchronize();
    printf("Calc complete\n");

    for(int i = 0; i < n; i++){
    if(z[i] != x[i] + y[i]){
        printf("ERROR: Expected %d, got %d\n", x[i]+y[i], z[i]);
    }
    }
    printf("\n");

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}
