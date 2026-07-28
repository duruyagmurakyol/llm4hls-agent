#include <cmath>
#include <iostream>
#include "atax.h"

namespace {
constexpr int M = 38;
constexpr int N = 42;
constexpr double TOLERANCE = 1e-9;

void initialise(double A[M][N], double x[N]) {
    for (int i = 0; i < N; ++i) x[i] = 1.0 + static_cast<double>(i) / static_cast<double>(N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            A[i][j] = static_cast<double>((i + j) % N) / (5.0 * M);
}

void reference_atax(const double A[M][N], const double x[N], double expected[N]) {
    double tmp[M] = {};
    for (int j = 0; j < N; ++j) expected[j] = 0.0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) tmp[i] += A[i][j] * x[j];
        for (int j = 0; j < N; ++j) expected[j] += A[i][j] * tmp[i];
    }
}
}

int main() {
    double A[M][N];
    double x[N];
    double y[N];
    double tmp[M];
    double expected[N];
    initialise(A, x);
    reference_atax(A, x, expected);
    kernel_atax(A, x, y, tmp);
    for (int i = 0; i < N; ++i) {
        if (std::fabs(y[i] - expected[i]) > TOLERANCE) {
            std::cerr << "FAIL index=" << i << " expected=" << expected[i] << " actual=" << y[i] << '\n';
            return 1;
        }
    }
    std::cout << "All ATAX tests passed.\n";
    return 0;
}
