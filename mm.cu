#include <stdio.h>


void init(float **a, float **b, float **c, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j<n; j++){
            a[i][j] = 0;
            b[i][j] = (i * j) % 16;
            c[i][j] = (i*j + 1)% 13;
        }
    }
}

void mm(float *a[], float *b[], float *c[], int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            for(int k = 0; k < n; k++){
                a[i][j] += b[i][k] * c[k][j];
            }
        }
    }
}

int main(){
    int n = 32;
    float a[n][n], b[n][n], c[n][n];

    init(a, b, c, n);
    mm(a, b, c, n);
    return 0;
}


