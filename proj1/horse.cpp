#include <stdio.h>

int main()
{
    int n = 32768;

    //regular
    float *m = malloc(n * n * sizeof(float));

    for(int i = 0; i < n * n; i++){
        m[i] = (i+5) % 13;
    }






    return 0;
}