#include <cstdio>
#include <chrono>

void get_histeq(int* hist, int* histeq, int total_num, bool printinfo=false) {
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
    if (printinfo)
      printf("%d\t", histeq[i]);
  }
}

void cpu_histeq(int* in_img, int* out_img, int size) {
  int hist[256] = {};
  for (int i = 0; i < size; ++i) {
    int pixel = in_img[i];
    hist[pixel]++;
  }
  int _map[256] = {0};
  get_histeq(hist, _map, size);
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

void print_array(int* array, int size, const char* name) {
  printf("\n%s:\n", name);
  for (int i = 0; i < size; ++i) {
    printf("%d\t", array[i]);
  }
  printf("\n");
}

template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}