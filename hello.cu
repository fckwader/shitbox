#include <stdio.h>





void add(int n, float *x, float *y)
{
    for(int i = 0; i < n; i++){
        z[i] = x[i] + y[i];
    }
}

int main()
{
    int n = 1000000;
    float x[n];
    float y[n];
    float z[n];

    //init
    for(int i = 0; i < n; i++){
        x[i] = (i * 2) % 13;
        y[i] = (i * 3) % 25;
        z[i] = 0;
    }

    add(n, x, y);

    for(int i = 0; i < n; i++){
        if(z[i] != x[i] + y[i]){
            printf("ERROR: Expected %d, got %d", x[i]+y[i], z[i]);
        }
    }

    return 0;
}
