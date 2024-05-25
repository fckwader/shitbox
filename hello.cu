#include <stdio.h>




__global__
void add(int n, float *x, float *y, float *z)
{
    for(int i = 0; i < n; i++){
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
    for(int i = 0; i < n; i++){
        x[i] = (i * 2) % 13;
        y[i] = (i * 3) % 25;
        z[i] = 0;
    }

    add<<<1, 1>>>(n, x, y, z);

    for(int i = 0; i < n; i++){
        if(z[i] != x[i] + y[i]){
            printf("ERROR: Expected %d, got %d", x[i]+y[i], z[i]);
        }
    }

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}
