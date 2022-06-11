__kernel void conv(__global int *data_in, __global int *data_out, int in_width,
                   int out_width, int item_size) {
  const int fil_size = 3;
  const int conv_kernel[3][3] = {{1, 1, 1}, {1, -9, 1}, {1, 1, 1}};
  int gid_x = get_global_id(0);
  int gid_y = get_global_id(1);
  int offset_x = gid_x * item_size;
  int offset_y = gid_y * item_size;

  for (int i = 0; i < item_size; i++) {
    for (int j = 0; j < item_size; j++) {
      int target_x = offset_x + i;
      int target_y = offset_y + j;
      if (target_x >= out_width || target_y >= out_width)
        continue;
      int sum = 0;
      for (int fi = 0; fi < fil_size; fi++) {
        for (int fj = 0; fj < fil_size; fj++) {
          int offset_2d = (target_x + fi) * in_width + target_y + fj;
          sum += data_in[offset_2d] * conv_kernel[fi][fj];
        }
      }
      data_out[target_x * out_width + target_y] = sum;
    }
  }
}
