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



void mm(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            for(int k = 0; k < n; k++){
                a[i*n +j] += b[i*n + k] * c[k*n + j];
            }
        }
    }
}

int main(){
    int n = 32;
    float a[n*n], b[n*n], c[n*n];

    init(a, b, c, n);
    mm(a, b, c, n);
    return 0;

}


