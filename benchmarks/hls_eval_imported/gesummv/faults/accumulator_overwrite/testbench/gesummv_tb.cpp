#include <cmath>
#include <cstdio>

#include "gesummv.h"

static void init_array(
    double *alpha,
    double *beta,
    double A[30][30],
    double B[30][30],
    double x[30]) {
    *alpha = 1.5;
    *beta = 1.2;
    for (int i = 0; i < 30; ++i) {
        x[i] = static_cast<double>(i) / 30.0;
        for (int j = 0; j < 30; ++j) {
            A[i][j] = static_cast<double>((i * j + 1) % 30) / 30.0;
            B[i][j] = static_cast<double>((i * j + 2) % 30) / 30.0;
        }
    }
}

static void reference_gesummv(
    double alpha,
    double beta,
    const double A[30][30],
    const double B[30][30],
    double tmp[30],
    const double x[30],
    double y[30]) {
    for (int i = 0; i < 30; ++i) {
        tmp[i] = 0.0;
        y[i] = 0.0;
        for (int j = 0; j < 30; ++j) {
            tmp[i] = A[i][j] * x[j] + tmp[i];
            y[i] = B[i][j] * x[j] + y[i];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
    }
}

int main() {
    double alpha;
    double beta;
    double A[30][30];
    double B[30][30];
    double tmp[30];
    double expected_tmp[30];
    double x[30];
    double y[30];
    double expected_y[30];

    init_array(&alpha, &beta, A, B, x);
    reference_gesummv(alpha, beta, A, B, expected_tmp, x, expected_y);
    kernel_gesummv(alpha, beta, A, B, tmp, x, y);

    for (int i = 0; i < 30; ++i) {
        if (std::fabs(tmp[i] - expected_tmp[i]) > 1e-8) {
            std::fprintf(stderr, "FAIL tmp[%d]: expected %.12f, got %.12f\n", i, expected_tmp[i], tmp[i]);
            return 1;
        }
        if (std::fabs(y[i] - expected_y[i]) > 1e-8) {
            std::fprintf(stderr, "FAIL y[%d]: expected %.12f, got %.12f\n", i, expected_y[i], y[i]);
            return 1;
        }
    }

    std::printf("All GESUMMV tests passed.\n");
    return 0;
}
