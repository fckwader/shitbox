#include <stdio.h>
#include <time.h>


void init(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j<n; j++){
            a[i*n + j] = 0;
            if(i == j){
                b[i*n +j] = 2;
            }else{
                b[i*n + j] = 0;
            }
            c[i*n + j] = (i* j + 1) % 7;
        }
    }
}

void mm_single(float *a, float *b, float *c, int n){
    for (int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            for(int k = 0; k < n; k++){
                a[i*n + j] += b[i*n + k] * c[k*n + j];
            }
        }
    }
}

__global__
void mm(float *a, float *b, float *c, int n){
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for(int i = tx; i < n; i+=blockDim.x)
    {
        for(int j = ty; j < n; j+=blockDim.y){
            for(int k = 0; k < n; k++){
                a[i * n + j] += b[i * n + k] * c[k * n + j];
            }
        }
    }
}

void printm(float *a, int n){
printf("-----\n");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%.0f ", a[i*n + j]);
        }
        printf("\n");
    }
printf("------\n");
}

int main(){
    int n = 2048;
    dim3 vec(32, 32, 1);
    float *a, *b, *c;
    cudaMallocManaged(&a, n*n*sizeof(float));
    cudaMallocManaged(&b, n*n*sizeof(float));
    cudaMallocManaged(&c, n*n*sizeof(float));

    init(a, b, c, n);
    clock_t start = clock();
        mm<<<1, vec>>>(a, b, c, n);
        cudaDeviceSynchronize();
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("MM parallel took %.1f seconds\n", time_spent);

    init(a, b, c, n);
    start = clock();
        mm_single(a, b, c, n);
        cudaDeviceSynchronize();
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("MM single took %.1f seconds\n", time_spent);


    return 0;

}


