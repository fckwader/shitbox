#include <stdio.h>


void init(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j<n; j++){
            a[i*n + j] = 0;
            b[i*n + j] = (i * j) % 16;
            c[i*n + j] = (i*j + 1)% 13;
        }
    }
}


__global__
void mm(float *a, float *b, float *c, int n){
    printf("X %d Y %d\n", threadIdx.x, threadIdx.y);
    return;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            for(int k = 0; k < n; k++){
                a[i*n +j] += b[i*n + k] * c[k*n + j];
            }
        }
    }
}

int main(){
    int n = 256;
    dim3 vec(8, 8, 1);
    float a[n*n], b[n*n], c[n*n];

    init(a, b, c, n);
    mm<<<1, vec>>>(a, b, c, n);
    cudaDeviceSynchronize();
    return 0;

}


