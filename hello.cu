#include <stdio.h>
#include <time.h>




__global__
void add(int n, float *x, float *y, float *z)
{


    int index = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    int stride = blockDim.x * blockDim.y * blockDim.z * gridDim.x;

    for (int i = index; i < n; i += stride){
        if(index == 0){
         //printf("0 0 is running i=%d\n", i);
        }
        z[i] = x[i] + y[i];
    }
}

void runBench(int n, float *x, float *y, float *z, int dimX, int dimY, int dimZ){
        dim3 sizevec(dimX, dimY, dimZ);
        printf("Calc...\n");
        clock_t begin = clock();
        add<<<1, sizevec>>>(n, x, y, z);
        cudaDeviceSynchronize();
        printf("Calc complete\n");
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("Time spent for %d %d %d: %f\n", dimX, dimY, dimZ, time_spent);

        printf("Verifying...\n");
        for(int i = 0; i < n; i++){
            if(z[i] != x[i] + y[i]){
                 printf("ERROR: Expected %d, got %d\n", x[i]+y[i], z[i]);
                 return -1;
            }
        }
        printf("Verified\n");

        printf("Resetting z...\n");
        for(int i = 0; i < n; i++){
            z[i] = 0;
        }
        printf("z reset\n");
}

int main()
{
    int n = 32768;
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

    runBench(n, x, y, z, 8, 8, 8);

    printf("\n");

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}
