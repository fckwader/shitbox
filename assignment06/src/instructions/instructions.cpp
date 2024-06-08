#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include <sched.h>
#include <unistd.h>

#ifndef OP
// arithmetic operator
#    define OP +
#endif

#ifndef CHAINS_MAX
// maximum number of dependency chains
#    define CHAINS_MAX 1U
#endif

#define STRINGIFY2(x) #x
#define STRINGIFY(x)  STRINGIFY2(x)

#define BENCH(T, C)   bench<T, C>(#T)

template <typename T, size_t CHAINS>
double instructions(size_t REP)
{
    T val1[CHAINS];
    T val2 = (rand() % REP) + 2.01;

    for (size_t j = 0; j < CHAINS; ++j)
        val1[j] = (rand() % (j + 1)) + 1 + rand();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (ssize_t i = 0; i < REP / CHAINS; ++i) {
#pragma GCC unroll 65534 // max loop unroll
        for (size_t j = 0; j < CHAINS; ++j) {
            val1[j] = val1[j] OP val2;
        }
        // prevent add optimization:
        val2 = (i > REP) ? 2.01 : 3.01;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    // prevent DCE:
    for (size_t j = 0; j < CHAINS; ++j)
        if (val1[j] == 1)
            std::cerr << "result: " << val1[j] << '\n';

    using dsec = std::chrono::duration<double>;
    return std::chrono::duration_cast<dsec>(t1 - t0).count();
}

template <typename T, size_t CHAINS>
void bench_chains(const char *typestring)
{
    // starting number of repetitions (must be divisible by CHAINS)
    size_t REP = (16000U / CHAINS) * CHAINS;

    constexpr double DURMIN = 0.5;
    double           dur    = 0.0;

    // warm-up and set REP to measure at least DURMIN seconds:
    while ((dur = instructions<T, CHAINS>(REP)) < DURMIN) {
        REP *= 2;
    }

    // actual measurement:
    dur = instructions<T, CHAINS>(REP);

    // csv output:
    std::cout << typestring << ','
              << STRINGIFY(OP) << ',' << CHAINS << ',' << dur / REP * 1.0e9 << '\n';
}

template <typename T, size_t CHAINS>
void bench(const char *typestring)
{
    bench_chains<T, CHAINS>(typestring);

    if constexpr (CHAINS > 1) {
        bench<T, CHAINS - 1>(typestring);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " R\n";
        std::cerr << "    R: random seed\n";
        return EXIT_FAILURE;
    }

    srand(atoi(argv[1]));

    // pinning on core 0:
    int       cpu = 0;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    sched_setaffinity(0, sizeof(cpu_set_t), &mask);

    std::cerr << "======================================\n"
              << "Running with OP=" << STRINGIFY(OP) 
              << " CHAINS_MAX=" << CHAINS_MAX << " \n"
              << "======================================\n";

    BENCH(float, CHAINS_MAX);
    BENCH(double, CHAINS_MAX);
    BENCH(int32_t, CHAINS_MAX);
    BENCH(int64_t, CHAINS_MAX);

    return EXIT_SUCCESS;
}
