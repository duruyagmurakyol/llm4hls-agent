#include <cmath>
#include <iostream>
#include "bicg.h"

namespace {
constexpr int N = 42;
constexpr int M = 38;
constexpr double TOLERANCE = 1e-9;

void initialise(double A[N][M], double p[M], double r[N]) {
    for (int j = 0; j < M; ++j) {
        p[j] = 1.0 + static_cast<double>(j) / static_cast<double>(M);
    }
    for (int i = 0; i < N; ++i) {
        r[i] = 1.0 + static_cast<double>(i) / static_cast<double>(N);
        for (int j = 0; j < M; ++j) {
            A[i][j] = static_cast<double>((i * j + 1) % N) / static_cast<double>(N);
        }
    }
}

void reference_bicg(
    const double A[N][M],
    const double p[M],
    const double r[N],
    double expected_s[M],
    double expected_q[N]) {
    for (int j = 0; j < M; ++j) {
        expected_s[j] = 0.0;
    }
    for (int i = 0; i < N; ++i) {
        expected_q[i] = 0.0;
        for (int j = 0; j < M; ++j) {
            expected_s[j] += r[i] * A[i][j];
            expected_q[i] += A[i][j] * p[j];
        }
    }
}
}

int main() {
    double A[N][M];
    double s[M];
    double q[N];
    double p[M];
    double r[N];
    double expected_s[M];
    double expected_q[N];

    initialise(A, p, r);
    reference_bicg(A, p, r, expected_s, expected_q);
    kernel_bicg(A, s, q, p, r);

    for (int j = 0; j < M; ++j) {
        if (std::fabs(s[j] - expected_s[j]) > TOLERANCE) {
            std::cerr << "FAIL s index=" << j << " expected=" << expected_s[j]
                      << " actual=" << s[j] << '\n';
            return 1;
        }
    }
    for (int i = 0; i < N; ++i) {
        if (std::fabs(q[i] - expected_q[i]) > TOLERANCE) {
            std::cerr << "FAIL q index=" << i << " expected=" << expected_q[i]
                      << " actual=" << q[i] << '\n';
            return 1;
        }
    }

    std::cout << "All BICG tests passed.\n";
    return 0;
}
