#pragma OPENCL EXTENSION cl_khr_global_uchar_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_uchar_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

// convert image to hist
__kernel void imgToHist(__global const int *imgMat, __global int *hist,
                        __local int *local_hist, int data_per_item,
                        __global int *log) {
  int l_idx = get_local_id(0);
  int g_idx = get_global_id(0);
  atomic_inc(log + l_idx);
  if (l_idx == 0)
    for (int i = 0; i < 256; i++)
      local_hist[i] = 0;

  int item_offset = g_idx * data_per_item; // 每个工作项处理的像素点位置偏移
  for (unsigned int i = item_offset; i < item_offset + data_per_item; i++) {
    // log[i] += 1;
    atomic_inc(local_hist + imgMat[i]);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (l_idx == 0) {
    for (int i = 0; i < 256; i++) {
      atomic_add(hist + i, local_hist[i]);
    }
  }
}


// 将均衡化的直方图用到图像上
__kernel void histEqToImg(__global int* imgMat, __global int* hist_eq, __global int* log)
{
	int g_idx = get_global_id(0);
  atomic_inc(log + g_idx);
	imgMat[g_idx] = hist_eq[imgMat[g_idx]];
}
