#include "cudautil.h"

#define BLOCK_SIZE 32

/*
 *TODO: Task b: Global memory MM implementation
 */
__global__ void globalMM(double *__restrict__ a,
                         double *__restrict__ b,
                         double *__restrict__ c,
                         int N,
                         int REP, double *flopcount)
{

    int    row = blockIdx.y * blockDim.y + threadIdx.y;
    int    col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    for (int r = 0; r < REP; ++r)
        if (col < N && row < N) {
            #pragma unroll
            for (int i = 0; i < N; i++) {
                sum += a[row * N + i] * b[i * N + col];
                //flopcount += 2;
            }

            c[row * N + col] = sum;
        }
}


/*
 *TODO: Task c: Implement Tiled Shared memory MM version
 */
__global__ void sharedTiledMM(double *__restrict__ a,
                              double *__restrict__ b,
                              double *__restrict__ c,
                              int N,
                              int REP)
{
    int tilesize = blockDim.x * blockDim.y;
    __shared__ double *ta, *tb;
    cudaMalloc(&ta, tilesize * sizeof(double));
    cudaMalloc(&tb, tilesize * sizeof(double));

    for(int i = 0; i < blockDim.x; i++){
        for(int j = 0; j < blockDim.y; j++){
            ta[i * blockDim.x + j] = a[i * blockDim.x + j];
            tb[i * blockDim.x + j] = b[i * blockDim.x + j];
            }
    }

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int column = blockIdx.y * blockDim.y + threadIdx.y;



    printf("Row: %d, Column: %d, Thread: %d %d\n", row, column, threadIdx.x, threadIdx.y);
}

int main(int argc, char *argv[])
{

    int device = 0;
    cudaSetDevice(device);

    if (argc < 2) {
        printf("For C(NxN) = A(NxN)* B(NxN), Matrix size value N must be provided !\n");
        exit(1);
    }

    char *pEnd;
    int   N = strtol(argv[1], &pEnd, 10);
    if (errno == ERANGE) {
        printf("Problem with the first number  N .");
        exit(2);
    }
    int          REP       = 1;
    unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3         dimGrid(grid_cols, grid_rows);
    dim3         dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    /* Memory allocations and initializations of matrices */
    double *a = (double *)malloc(sizeof(double) * N * N);
    double *b = (double *)malloc(sizeof(double) * N * N);
    double *c = (double *)malloc(sizeof(double) * N * N);

    double *flopcount;
    cudaMallocManaged(&flopcount, sizeof(double));

    double *d_a, *d_b, *d_c;
    /*
     * TODO:Task e: Use UVA for device memory
     */
    cudaMalloc(&d_a, sizeof(double) * N * N);
    cudaMalloc(&d_b, sizeof(double) * N * N);
    cudaMalloc(&d_c, sizeof(double) * N * N);

    // Initialization on CPU
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N * N; ++i) {
        a[i] = atan(i);
        b[i] = cos(i);
        c[i] = 0.0;
    }
    // Copy initial values to GPUs
    cudaMemcpy(d_a, a, sizeof(double) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(double) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, sizeof(double) * N * N, cudaMemcpyHostToDevice);

    using dsec = std::chrono::duration<double>;
    double gf  = 2.0 * (double)N * N * N * REP * 1.0e-9;

    // Compute Checksum for Simple Correctness Checks
    double checksum = cpu_matrix_mult_checksum(a, b, N, REP);

    /*
     * Basic MM Kernel Call & Time Measurements
     */
    dim3 dimBlockMM(N, N);
    auto t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK_ERR_LAST(globalMM<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N, REP, flopcount));
    CUDA_CHECK_ERR(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    // Calculate Flops/sec,
    double dur = std::chrono::duration_cast<dsec>(t1 - t0).count();
    std::cout << "MM GFlops/s (N=" << N << "): " << gf / dur << std::endl;

    // Copy the result back to CPU & correctness check
    cudaMemcpy(c, d_c, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
    Checksum(N, c, checksum);

    // Reset result_arrays c and d_c
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N * N; ++i) {
        c[i] = 0.0;
    }
    cudaMemcpy(d_c, c, sizeof(double) * N * N, cudaMemcpyHostToDevice);


    t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK_ERR_LAST(sharedTiledMM<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N, REP));
    CUDA_CHECK_ERR(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();

    // Calculate Flops/sec, Correctness Checks & Reset Result array C
    dur = std::chrono::duration_cast<dsec>(t1 - t0).count();
    std::cout << "Shared Tiled GFlops/s (N=" << N << "): " << gf / dur << std::endl;

    // Copy the result back to CPU & correctness check
    cudaMemcpy(c, d_c, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
    Checksum(N, c, checksum);


  //  printf("gf: %f\n", gf);
 //   printf("Flop count: %f\n", flopcount);

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
