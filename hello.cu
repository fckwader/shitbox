#include <stdio.h>

__global__
void print()
{
    printf("FAWKKK");
}

int main()
{
    print<<<1, 1>>>();
}
