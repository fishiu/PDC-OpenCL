__kernel void conv(const __global int *data_in, __global int *data_out,
                   const int in_width, const int out_width, const int item_size,
                   __constant int *filter, const int fil_size) {
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
      // for (int fi = 0; fi < fil_size; fi++) {
      //   for (int fj = 0; fj < fil_size; fj++) {
      //     int offset_2d = (target_x + fi) * in_width + target_y + fj;
      //     sum += data_in[offset_2d] * filter[fi * fil_size + fj];
      //   }
      // }
      // unrole loop
      sum += data_in[(target_x + 0) * in_width + target_y + 0] * filter[0 * fil_size + 0];
      sum += data_in[(target_x + 0) * in_width + target_y + 1] * filter[0 * fil_size + 1];
      sum += data_in[(target_x + 0) * in_width + target_y + 2] * filter[0 * fil_size + 2];
      sum += data_in[(target_x + 1) * in_width + target_y + 0] * filter[1 * fil_size + 0];
      sum += data_in[(target_x + 1) * in_width + target_y + 1] * filter[1 * fil_size + 1];
      sum += data_in[(target_x + 1) * in_width + target_y + 2] * filter[1 * fil_size + 2];
      sum += data_in[(target_x + 2) * in_width + target_y + 0] * filter[2 * fil_size + 0];
      sum += data_in[(target_x + 2) * in_width + target_y + 1] * filter[2 * fil_size + 1];
      sum += data_in[(target_x + 2) * in_width + target_y + 2] * filter[2 * fil_size + 2];
      data_out[target_x * out_width + target_y] = sum;
    }
  }
}
