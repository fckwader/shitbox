#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    int n = 50000;

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

    int tilesize = 8;

    start = clock();
    for(int i = 0; i < n; i += tilesize){
        for(int j = 0; j < n; j += tilesize){

            for(int ti = i; ti < i + tilesize && ti < n; ti++){
                for(int tj = j; tj < j + tilesize && tj< n; tj++){
                        m[ti*n + tj] += 15;
                }
            }


        }
    }
    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Tiled style took %.2f seconds\n", seconds);



    return 0;
}