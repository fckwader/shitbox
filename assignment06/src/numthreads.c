#include <stdio.h>
#include <omp.h>

int main(){

    #pragma omp parallel
    {
        printf("%d\n", OMP_GET_NUM_THREADS());
    }


    return 0;
}