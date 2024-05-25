#include <stdio.h>




__global__
void add(int n, float *x, float *y, float *z)
{
    printf("Calc...\n");
    for(int i = 0; i < n; i++){
        z[i] = x[i] + y[i];
    }
    printf("Calc complete\n");
}

int main()
{
    int n = 1000000000;
    float *x, *y, *z;
    cudaMallocManaged(&x, n*sizeof(float));
    cudaMallocManaged(&y, n*sizeof(float));
    cudaMallocManaged(&z, n*sizeof(float));

    //init
    printf("Init...\n");
    for(int i = 0; i < n; i++){
        x[i] = (i * 2) % 13;
        y[i] = (i * 3) % 25;
        z[i] = 0;
    }
    printf("Init complete\n");

    add<<<1, 256>>>(n, x, y, z);

    cudaDeviceSynchronize();

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
