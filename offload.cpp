#include <stdio.h>

int main(){

    #pragma omp target
    {
        int i = 0;
        i += 1;
    }


    return 0;
}