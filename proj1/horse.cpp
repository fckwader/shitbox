#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    int n = 32768;

    double *m = (double *) malloc(n * n * sizeof(double));

    for(int i = 0; i < n * n; i++){
        m[i] = (i+5) % 13;
    }

    //regular style
    clock_t start = clock();
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            m[i*n + j] += 15;
        }
    }
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Regular style took %.2f seconds\n", seconds);



    return 0;
}