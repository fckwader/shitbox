#include <stdio.h>
#include <time.h>




__global__
void add(int n, float *x, float *y, float *z)
{
    int index = blockIdx.x * blockDim.x * blockDim.y * blockDim.z + threadIdx.x;
    int stride = blockDim.x * blockDim.y * blockDim.z * gridDim.x;

    for (int i = index; i < n; i += stride){
        z[i] = x[i] + y[i];
    }
}

int main()
{
    int n = 32768;
    int blockSize = 8;
    dim3 sizevec(blockSize, blockSize, blockSize);
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


    printf("Calc...\n");
    clock_t begin = clock();
    add<<<1, sizevec>>>(n, x, y, z);
    cudaDeviceSynchronize();
    printf("Calc complete\n");
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent: %f\n", time_spent);

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
