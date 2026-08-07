#include <cmath>
#include <cstdio>

#include "gemver.h"

static void init_array(
    double *alpha,
    double *beta,
    double A[40][40],
    double u1[40],
    double v1[40],
    double u2[40],
    double v2[40],
    double w[40],
    double x[40],
    double y[40],
    double z[40]) {
    *alpha = 1.5;
    *beta = 1.2;
    const double n = 40.0;

    for (int i = 0; i < 40; ++i) {
        u1[i] = i;
        u2[i] = ((i + 1) / n) / 2.0;
        v1[i] = ((i + 1) / n) / 4.0;
        v2[i] = ((i + 1) / n) / 6.0;
        y[i] = ((i + 1) / n) / 8.0;
        z[i] = ((i + 1) / n) / 9.0;
        x[i] = 0.0;
        w[i] = 0.0;
        for (int j = 0; j < 40; ++j)
            A[i][j] = static_cast<double>((i * j + 3 * i + j) % 40) / 40.0;
    }
}

static void reference_gemver(
    double alpha,
    double beta,
    double A[40][40],
    const double u1[40],
    const double v1[40],
    const double u2[40],
    const double v2[40],
    double w[40],
    double x[40],
    const double y[40],
    const double z[40]) {
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j)
            A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j)
            x[i] = x[i] + beta * A[j][i] * y[j];

    for (int i = 0; i < 40; ++i)
        x[i] = x[i] + z[i];

    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j)
            w[i] = w[i] + alpha * A[i][j] * x[j];
}

static bool close(double actual, double expected) {
    return std::fabs(actual - expected) <= 1e-8;
}

int main() {
    double alpha;
    double beta;
    double A[40][40];
    double expected_A[40][40];
    double u1[40];
    double v1[40];
    double u2[40];
    double v2[40];
    double w[40];
    double expected_w[40];
    double x[40];
    double expected_x[40];
    double y[40];
    double z[40];

    init_array(&alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
    for (int i = 0; i < 40; ++i) {
        expected_w[i] = w[i];
        expected_x[i] = x[i];
        for (int j = 0; j < 40; ++j)
            expected_A[i][j] = A[i][j];
    }

    reference_gemver(
        alpha, beta, expected_A, u1, v1, u2, v2,
        expected_w, expected_x, y, z
    );
    kernel_gemver(alpha, beta, A, u1, v1, u2, v2, w, x, y, z);

    for (int i = 0; i < 40; ++i) {
        if (!close(x[i], expected_x[i])) {
            std::fprintf(stderr, "FAIL x[%d]: expected %.12f, got %.12f\n", i, expected_x[i], x[i]);
            return 1;
        }
        if (!close(w[i], expected_w[i])) {
            std::fprintf(stderr, "FAIL w[%d]: expected %.12f, got %.12f\n", i, expected_w[i], w[i]);
            return 1;
        }
        for (int j = 0; j < 40; ++j) {
            if (!close(A[i][j], expected_A[i][j])) {
                std::fprintf(stderr, "FAIL A[%d][%d]\n", i, j);
                return 1;
            }
        }
    }

    std::printf("All GEMVER tests passed.\n");
    return 0;
}
