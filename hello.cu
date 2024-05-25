#include <stdio.h>

__global__
int print()
{
    printf("FAWKKK");
}

int main()
{
    print<<<1, 1>>>();
}
