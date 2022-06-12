__kernel void conv(const __global int *imgin, __global int *imgout,
                   const int outw, __constant int *filter, const int fil_size,
                   __local int *local_img) {
  const int inw = get_global_size(0);
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);

  const int local_size = get_local_size(0);  // local work item size
  const int overlap = inw - outw;
  const int inw_l = local_size + overlap; // plus overlap
  const int lid_x = get_local_id(0);
  const int lid_y = get_local_id(1);
  // initialize local memory and sync
  if (lid_x == 0 && lid_y == 0)
    for (int i = 0; i < inw_l; ++i)
      for (int j = 0; j < inw_l; ++j) {
        if ((gid_x + i) >= inw || (gid_y + j) >= inw) continue;
        local_img[i * inw_l + j] = imgin[(gid_x + i) * inw + (gid_y + j)];
      }
  barrier(CLK_LOCAL_MEM_FENCE);

  // do not need to compute here
  if (gid_x >= outw || gid_y >= outw)
    return;

  int sum = 0;
  // unrole loop
  sum += local_img[lid_x * inw_l + lid_y] * filter[0];
  sum += local_img[lid_x * inw_l + lid_y + 1] * filter[1];
  sum += local_img[lid_x * inw_l + lid_y + 2] * filter[2];
  sum += local_img[(lid_x + 1) * inw_l + lid_y] * filter[fil_size];
  sum += local_img[(lid_x + 1) * inw_l + lid_y + 1] * filter[fil_size + 1];
  sum += local_img[(lid_x + 1) * inw_l + lid_y + 2] * filter[fil_size + 2];
  sum += local_img[(lid_x + 2) * inw_l + lid_y] * filter[2 * fil_size];
  sum += local_img[(lid_x + 2) * inw_l + lid_y + 1] * filter[2 * fil_size + 1];
  sum += local_img[(lid_x + 2) * inw_l + lid_y + 2] * filter[2 * fil_size + 2];
  imgout[gid_x * outw + gid_y] = sum;
}
