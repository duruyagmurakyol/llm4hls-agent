#include <cmath>
#include <cstdio>

#include "gemm.h"

static void init_array(
    double *alpha,
    double *beta,
    double C[20][25],
    double A[20][30],
    double B[30][25]) {

    *alpha = 1.5;
    *beta = 1.2;

    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 25; ++j)
            C[i][j] = static_cast<double>((i * j + 1) % 20) / 20.0;

    for (int i = 0; i < 20; ++i)
        for (int k = 0; k < 30; ++k)
            A[i][k] = static_cast<double>(i * (k + 1) % 30) / 30.0;

    for (int k = 0; k < 30; ++k)
        for (int j = 0; j < 25; ++j)
            B[k][j] = static_cast<double>(k * (j + 2) % 25) / 25.0;
}

static void reference_gemm(
    double alpha,
    double beta,
    double C[20][25],
    const double A[20][30],
    const double B[30][25]) {

    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 25; ++j)
            C[i][j] *= beta;

        for (int k = 0; k < 30; ++k)
            for (int j = 0; j < 25; ++j)
                C[i][j] += alpha * A[i][k] * B[k][j];
    }
}

int main() {
    double alpha;
    double beta;
    double C[20][25];
    double expected[20][25];
    double A[20][30];
    double B[30][25];

    init_array(&alpha, &beta, C, A, B);

    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 25; ++j)
            expected[i][j] = C[i][j];

    reference_gemm(alpha, beta, expected, A, B);
    kernel_gemm(alpha, beta, C, A, B);

    constexpr double tolerance = 1e-8;

    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 25; ++j) {
            const double error = std::fabs(C[i][j] - expected[i][j]);

            if (error > tolerance) {
                std::fprintf(
                    stderr,
                    "Mismatch at C[%d][%d]: expected %.12f, got %.12f, "
                    "error %.12e\n",
                    i,
                    j,
                    expected[i][j],
                    C[i][j],
                    error
                );
                return 1;
            }
        }
    }

    std::printf("All GEMM tests passed.\n");
    return 0;
}
