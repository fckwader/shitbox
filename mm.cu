#include <stdio.h>


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


__global__
void mm(float *a, float *b, float *c, int n){
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    printf("TX %d, TY %d\n", tx, ty);

    for(int j = ty; j < n; j+=blockDim.y){
        for(int k = 0; k < n; k++){
            if(tx == 0 && ty == 0){
               printf("j %d\n", j);
            }
            a[tx * n + j] += b[tx * n + k] * c[k * n + j];
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
    int n = 10;
    dim3 vec(8, 8, 1);
    float a[n*n], b[n*n], c[n*n];

    init(a, b, c, n);
    printm(c, n);
    mm<<<1, vec>>>(a, b, c, n);
    cudaDeviceSynchronize();
    printm(a, n);
    return 0;

}


