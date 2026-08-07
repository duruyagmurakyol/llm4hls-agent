#include <cmath>
#include <cstdio>

#include "syrk.h"

static void init_array(
    double *alpha,
    double *beta,
    double C[30][30],
    double A[30][20]) {
    *alpha = 1.5;
    *beta = 1.2;
    for (int i = 0; i < 30; ++i)
        for (int j = 0; j < 20; ++j)
            A[i][j] = static_cast<double>((i * j + 1) % 30) / 30.0;
    for (int i = 0; i < 30; ++i)
        for (int j = 0; j < 30; ++j)
            C[i][j] = static_cast<double>((i * j + 2) % 20) / 20.0;
}

static void reference_syrk(
    double alpha,
    double beta,
    double C[30][30],
    const double A[30][20]) {
    for (int i = 0; i < 30; ++i) {
        for (int j = 0; j <= i; ++j)
            C[i][j] *= beta;
        for (int k = 0; k < 20; ++k)
            for (int j = 0; j <= i; ++j)
                C[i][j] += alpha * A[i][k] * A[j][k];
    }
}

int main() {
    double alpha;
    double beta;
    double C[30][30];
    double expected_C[30][30];
    double A[30][20];

    init_array(&alpha, &beta, C, A);
    for (int i = 0; i < 30; ++i)
        for (int j = 0; j < 30; ++j)
            expected_C[i][j] = C[i][j];

    reference_syrk(alpha, beta, expected_C, A);
    kernel_syrk(alpha, beta, C, A);

    for (int i = 0; i < 30; ++i) {
        for (int j = 0; j < 30; ++j) {
            if (std::fabs(C[i][j] - expected_C[i][j]) > 1e-8) {
                std::fprintf(stderr, "FAIL C[%d][%d]: expected %.12f, got %.12f\n", i, j, expected_C[i][j], C[i][j]);
                return 1;
            }
        }
    }

    std::printf("All SYRK tests passed.\n");
    return 0;
}
