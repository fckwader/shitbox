#include <stdio.h>
#include <stdlib.h>

int main()
{
    int n = 32768;

    //regular
    double *m = malloc(n * n * sizeof(double));

    for(int i = 0; i < n * n; i++){
        m[i] = (i+5) % 13;
    }






    return 0;
}