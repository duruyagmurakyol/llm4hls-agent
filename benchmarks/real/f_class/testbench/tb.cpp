#include <cstdio>
#include "kernel.h"
int main() {
  int a[N], b[N], out[N];
  for (int i = 0; i < N; ++i) { a[i] = i; b[i] = 2 * i; }
  vadd(a, b, out);
  int errors = 0;
  for (int i = 0; i < N; ++i) if (out[i] != 3 * i) ++errors;
  if (errors) { printf("FAIL %d\n", errors); return 1; }
  printf("PASS\n");
  return 0;
}
