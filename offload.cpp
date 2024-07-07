#include <stdio.h>

int main(){

    #pragma omp target
    {
        printf("hello\n");
    }


    return 0;
}