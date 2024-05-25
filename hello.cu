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
        printf("i = %d\n", i);
        x[i] = (float) i;
        y[i] = (float) i;
        z[i] = (float) 0;
        printf("INIT X: %d, Y: %d, Z: %d\n", x[i], y[i], z[i]);
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
     printf("%d + %d = %d\n", x[i], y[i], z[i]);
    }
    printf("\n");

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}
