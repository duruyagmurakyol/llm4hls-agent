#include <cmath>
#include <cstdio>

#include "mvt.h"

static void init_array(
    double x1[40],
    double x2[40],
    double y1[40],
    double y2[40],
    double A[40][40]) {
    for (int i = 0; i < 40; ++i) {
        x1[i] = static_cast<double>(i) / 40.0;
        x2[i] = static_cast<double>(i + 1) / 40.0;
        y1[i] = static_cast<double>(i + 3) / 40.0;
        y2[i] = static_cast<double>(i + 4) / 40.0;
        for (int j = 0; j < 40; ++j)
            A[i][j] = static_cast<double>((i * (j + 1) + 2 * j + 1) % 40) / 40.0;
    }
}

static void reference_mvt(
    double x1[40],
    double x2[40],
    const double y1[40],
    const double y2[40],
    const double A[40][40]) {
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j)
            x1[i] = x1[i] + A[i][j] * y1[j];

    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j)
            x2[i] = x2[i] + A[j][i] * y2[j];
}

int main() {
    double x1[40];
    double expected_x1[40];
    double x2[40];
    double expected_x2[40];
    double y1[40];
    double y2[40];
    double A[40][40];

    init_array(x1, x2, y1, y2, A);
    for (int i = 0; i < 40; ++i) {
        expected_x1[i] = x1[i];
        expected_x2[i] = x2[i];
    }

    reference_mvt(expected_x1, expected_x2, y1, y2, A);
    kernel_mvt(x1, x2, y1, y2, A);

    for (int i = 0; i < 40; ++i) {
        if (std::fabs(x1[i] - expected_x1[i]) > 1e-8) {
            std::fprintf(stderr, "FAIL x1[%d]: expected %.12f, got %.12f\n", i, expected_x1[i], x1[i]);
            return 1;
        }
        if (std::fabs(x2[i] - expected_x2[i]) > 1e-8) {
            std::fprintf(stderr, "FAIL x2[%d]: expected %.12f, got %.12f\n", i, expected_x2[i], x2[i]);
            return 1;
        }
    }

    std::printf("All MVT tests passed.\n");
    return 0;
}
