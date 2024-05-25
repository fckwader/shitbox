#include <stdio.h>




__global__
void add(int n, float *x, float *y, float *z)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for(int i = index; i < n; i += stride){
        z[i] = x[i] + y[i];
    }
}

int main()
{
    int n = 8;
    float *x, *y, *z;
    cudaMallocManaged(&x, n*sizeof(float));
    cudaMallocManaged(&y, n*sizeof(float));
    cudaMallocManaged(&z, n*sizeof(float));

    //init
    printf("Init...\n");
    for(int i = 0; i < n; i++){
        x[i] = (i * 2.5) % 13;
        y[i] = (i * 1.9) % 25;
        z[i] = 0;
    }
    printf("Init complete\n");

    printf("Calc...\n");
    add<<<1, 256>>>(n, x, y, z);
    cudaDeviceSynchronize();
    printf("Calc complete\n");

    for(int i = 0; i < n; i++){
       // if(z[i] != x[i] + y[i]){
     //       printf("ERROR: Expected %d, got %d\n", x[i]+y[i], z[i]);
     //   }
     printf("%f + %f = %f\n", x[i], y[i], z[i]);
    }
    printf("\n");

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}
