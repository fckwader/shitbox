#include <stdio.h>

int main()
{

    #pragma omp parallel
    {
        #pragma omp for nowait
        for(int i = 0; i < 10; i++)
        {
            printf("%d ", i);
        }
    }

    printf("\n");
    return 0;

}