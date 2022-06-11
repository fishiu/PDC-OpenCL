# PDC 作业3 报告

信息管理系 金笑缘

---

## 〇 写在前面

本实验作业通过两个图像相关算法的并行化来探索OpenCL并行框架。回顾整个实验过程，相比前两次作业，OpenCL花费了更多时间用来配置环境，并且代码编写的过程也更加痛苦，一方面可能因为没有在课程中通过实践来探索OpenCL的使用方法，另一方面OpenCL确实比OpenMP和MPI的业务代码更加底层，更加复杂。

### 1 关于OpenCL

经过这次实验我基本对OpenCL有了大致的了解，个人感觉它的主要特点有：

- 异构平台的兼容，尽管实验仅仅使用了GPU，没有测试FPGA和CPU进行测试；
- 通过Kernel函数来实现并行化编程，这种基于Kernel的框架更加方便编写灵活的代码，但是越灵活就说明封装越少，C++中的业务代码就显得非常麻烦；
- 另外Kernel中的原子化操作atomxxx比较有用；
- 和其他框架不同的是OpenCL提供了对硬件的接口：存储模型。也就是通过Global、Local、Private等来指定存储模型，这样的存储模型使得可以在OpenCL中使用硬件特点来实现并行化编程。

也有一些缺点：

- debug很麻烦，我用了非常原始的方法来解决debug的问题，在后面进一步介绍；
- 报错处理也很麻烦，每一个CL操作都会返回一个errNum，需要监视这个变量对我来说比较麻烦。

### 2 实验探索路径

由于课件中似乎没有OpenCL的Hello World示例程序，除了服务器上的pdc_test之外我还找了如下教程，在此处分享：
- [Hands On OpenCL](http://handsonopencl.github.io/)
- [UL: HPC-Team Tutorial](https://ulhpc-tutorials.readthedocs.io/en/latest/gpu/opencl/)
- [OpenCL 内存性能优化](https://zhuanlan.zhihu.com/p/396912769)
- [OpenCL 工作组性能优化](https://blog.csdn.net/weixin_38498942/article/details/116993722)

---

## 一 直方图均衡化

### 1 实验目标

直方图均衡化是一种图片亮度算法，本质上分为三个阶段：
1. 直方图分布计算（统计各个像素点）
2. 直方图均衡化（针对直方图本身的计算）
3. 图片的映射（点对点的计算）

整个过程其实和图片的二维结构完全无关，因此这就是个一维的并行算法（维度在OpenCL中是一个重要的概念）

### 2 实验设计

主要考虑上述三个阶段的并行化
1. 像素点统计，这里会涉及到频繁的对hist映射数组的读写（特别是写的过程很容易产生冲突），如果把它作为global变量可能会比较慢，这里我设计的优化方法是每一个工作组都建立一个local hist作为缓存，同一个工作组内的所有工作项在的读写工作在local hist内存上进行，全部完成后使用`barrier(CLK_GLOBAL_MEM_FENCE);`进行同步。
2. 直方图本身的变换，这是一个根据pdf计算cdf的过程，数组的大小为256，因此我认为根本无需并行化：
  - 256太小了，相比图片尺寸无足挂齿
  - cdf的计算过程是一个线性的过程，因此很难高效的并行化
3. 像素点映射，这里只涉及对hist_eq变量的读操作，因此我没有使用local内存，简单的将每一个像素作为一个工作项。

具体的Kernel代码如下：
```opencl
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
```
### 3 实验结果

固定随机种子为1107（我的生日），为每一个像素点随机生成一个[0, 256)范围内的像素值。使用cpu算法和gpu算法，核对结果准确性后比较使用的时间，同时测试不同图片尺寸的计算时间：

### 4 实验结论


---

## 二 图像锐化

### 1 实验目标

使用卷积进行锐化本质是一个二维矩阵的卷积，这里相比第一题确实需要用到维度为2来进行并行化。

### 2 实验设计

卷积操作没有复杂的步骤，难点在于如何进行数据的划分，本实验中我从两个维度对需要计算的任务矩阵进行划分，每一个工作项负责处理大小为`item_size`的矩阵任务目标，多个工作项共同完成`out_width * out_width`个任务。

内存优化需要对输入的图片数据进行local或private，以及对滤波器参数进行优化。本实验由于时间精力问题没有对输入图片进行优化，因此我们只对滤波器参数进行优化，因此我们使用一个private内存来存储滤波器参数，这样每一个工作项都可以快速滤波器内存，提高并行效率。

我认为本实验的工作项大小也是非常重要的因素，因此对工作项大小参数进行简单的探索。

```opencl
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
```

### 3 实验结果

固定随机种子为1107（我的生日），为每一个像素点随机生成一个[0, 256)范围内的像素值。使用cpu算法和gpu算法，核对结果准确性后比较使用的时间，同时测试不同图片尺寸的计算时间：

### 4 实验结论

- 考虑针对特定卷积核的优化
- OpenCL的内存模型还不够熟练，比如哪些数据应该放在哪里我还不太懂
- 数据划分