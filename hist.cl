__kernel void img_to_hist(__global const int *img, __global int *hist,
                          __local int *local_hist, int item_size) {
  int lid = get_local_id(0);
  int gid = get_global_id(0);
  // initialize local hist
  if (lid == 0)
    for (int i = 0; i < 256; i++)
      local_hist[i] = 0;
  barrier(CLK_GLOBAL_MEM_FENCE);

  int offset = gid * item_size;
  for (int i = offset; i < offset + item_size; i++)
    atomic_inc(local_hist + img[i]);

  // synchronize to reduce hist
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (lid == 0)
    for (int i = 0; i < 256; i++)
      atomic_add(hist + i, local_hist[i]);
}

__kernel void eq_img(__global int *img, __global int *hist_eq) {
  int gid = get_global_id(0);
  img[gid] = hist_eq[img[gid]];
}
