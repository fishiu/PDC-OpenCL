#include <cstdio>

void cpu_histeq(int* in_img, int* out_img, int size) {
  int hist[256] = {};
  for (int i = 0; i < size; ++i) {
    int pixel = in_img[i];
    hist[pixel]++;
  }
  float pdf[256] = {};
  for (int i = 0; i < 256; ++i) {
    pdf[i] = hist[i] * 1.0 / size;
  }
  float cdf[256] = {0};
  for (int i = 0; i < 256; ++i) {
    if (i == 0)
      cdf[i] = pdf[i];
    else
      cdf[i] = cdf[i - 1] + pdf[i];
  }
  int _map[256] = {0};
  printf("\ncpu_histeq:\n");
  for (int i = 0; i < 256; ++i) {
    _map[i] = (int)(255.0 * cdf[i] + 0.5);
    printf("%d\t", _map[i]);
  }
  printf("\n");
  for (int i = 0; i < size; ++i) {
    int pixel = in_img[i];
    out_img[i] = _map[pixel];
  }
}

bool compare(int* cpu_img, int* gpu_img, int size) {
  for (int i = 0; i < size; ++i) {
    if ((int)cpu_img[i] != gpu_img[i]) {
      printf("%d %d != %d\n", i, cpu_img[i], gpu_img[i]);
      return false;
    }
  }
  return true;
}

void get_histeq(int* hist, int* histeq, int total_num) {
  float pdf[256] = {};
  for (int i = 0; i < 256; ++i)
    pdf[i] = hist[i] * 1.0 / total_num;

  float cdf[256] = {0};
  for (int i = 0; i < 256; ++i) {
    if (i == 0)
      cdf[i] = pdf[i];
    else
      cdf[i] = cdf[i - 1] + pdf[i];
  }

  for (int i = 0; i < 256; ++i) {
    histeq[i] = (int)(255.0 * cdf[i] + 0.5);
    printf("%d\t", histeq[i]);
  }
}