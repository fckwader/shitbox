#include <stdio.h>

int n = 1000000;
float x[n];
float y[n];

//init
for(int i = 0; i < n; i++){
    x[i] = (i * 2) % 13;
    y[i] = (i * 3) % 25;
}

void add(int n, float *x, float *y)
{
    for(int i = 0; i < n; i++){
        x[i] += y[i];
    }
}

int main()
{
    add(n, x, y);
    return 0;
}
