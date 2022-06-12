__kernel void conv(const __global int *imgin, __global int *imgout,
                   const int outw, __constant int *filter, const int fil_size) {
  const int inw = get_global_size(0);
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);
  if (gid_x >= outw || gid_y >= outw) return;
  
  int sum = 0;
  // unrole loop
  sum += imgin[(gid_x)*inw + gid_y] * filter[0];
  sum += imgin[(gid_x)*inw + gid_y + 1] * filter[1];
  sum += imgin[(gid_x)*inw + gid_y + 2] * filter[2];
  sum += imgin[(gid_x + 1) * inw + gid_y] * filter[1 * fil_size];
  sum += imgin[(gid_x + 1) * inw + gid_y + 1] * filter[1 * fil_size + 1];
  sum += imgin[(gid_x + 1) * inw + gid_y + 2] * filter[1 * fil_size + 2];
  sum += imgin[(gid_x + 2) * inw + gid_y] * filter[2 * fil_size];
  sum += imgin[(gid_x + 2) * inw + gid_y + 1] * filter[2 * fil_size + 1];
  sum += imgin[(gid_x + 2) * inw + gid_y + 2] * filter[2 * fil_size + 2];
  imgout[gid_x * outw + gid_y] = sum;
}
