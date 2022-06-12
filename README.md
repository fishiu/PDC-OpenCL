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

| Picture width | OpenCL (ms)                | CPU (ms) |
| ------------- | -------------------------- | -------- |
| 512           | 241                        | 1        |
| 1024          | 248                        | 7        |
| 2048          | 272                        | 28       |
| 4096          | 298                        | 110      |
| 8192          | 458                        | 432      |

### 4 实验结论


---

## 二 图像锐化

### 1 实验目标

使用卷积进行锐化本质是一个二维矩阵的卷积，这里相比第一题确实需要用到维度为2来进行并行化。

### 2 实验设计

卷积操作没有复杂的步骤，难点在于如何进行数据或者任务的划分，本实验中我从图片的两个维度对需要计算的任务矩阵进行划分，每一个工作项负责处理大小为`item_size`的矩阵任务目标，多个工作项共同完成`out_width * out_width`个任务。（在最后的优化中，我改变了这个设定）。

内存优化需要对输入的图片数据进行local或private，以及对滤波器参数进行优化。首先只对滤波器参数进行优化，因此最初我使用一个private内存来存储滤波器参数，这样每一个工作项都可以快速滤波器内存，提高并行效率。但是后来考虑到在每一个工作项中可能会占据较大的开销，放入constant内存显然是更好的。

```opencl
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
      for (int fi = 0; fi < fil_size; fi++) {
        for (int fj = 0; fj < fil_size; fj++) {
          int offset_2d = (target_x + fi) * in_width + target_y + fj;
          sum += data_in[offset_2d] * filter[fi * fil_size + fj];
        }
      }
      data_out[target_x * out_width + target_y] = sum;
    }
  }
}
```

之后又实现了三个改进，具体内容见第4节。

实验固定随机种子为1107（我的生日），为每一个像素点随机生成一个[0, 256)范围内的像素值。使用cpu算法和gpu算法，核对结果准确性后比较使用的时间，同时测试不同图片尺寸的计算时间。

### 3 优化和结果

#### 优化1：constant filter

考虑到如果每个工作项都存储一个滤波器，开销会比较大，因此改为了使用constant内存进行存储filter（主要工作就是把filter作为一维数组输入kernel）。

| Picture width | OpenCL - constant filter (ms) | CPU (ms) |
| ------------- | -------------------------- | -------- |
| 256           | 249                        | 2        |
| 512           | 254                        | 11       |
| 1024          | 273                        | 44       |
| 2048          | 326                        | 177      |
| 4096          | 578                        | 704      |

local item size=8

#### 优化2：展开滤波器循环

GPU是厌恶循环的（相比CPU），因此考虑把循环展开，部分代码如下所示：

```opencl
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
```

展开循环后使用该kernel的实验结果如下表所示，实验所用的item size为8

| Picture width | OpenCL - flatten filter (ms) | CPU (ms) |
| ------------- | -------------------------- | -------- |
| 256           | 260                        | 2        |
| 512           | 248                        | 11       |
| 1024          | 268                        | 44       |
| 2048          | 321                        | 177      |
| 4096          | 588                        | 704      |

#### 优化3：使用local memory

根据经验，工作组大小也是非常重要的因素，因此考虑对工作组大小参数进行探索，首先要做的肯定是建立local memory，否则工作项数量对性能应该影响不大。但是由于我之前的设置就是每个工作项计算多个像素点，在这样的基础上，local memory的初始化等工作变得非常冗长，因此决定推倒重来：改为每个工作项只计算一个像素。

这个改进是简单的，结果如下所示：

| Picture width | OpenCL - naive (ms) | CPU (ms) |
| ------------- | -------------------------- | -------- |
| 256           | 252                        | 2        |
| 512           | 253                        | 11       |
| 1024          | 270                        | 44       |
| 2048          | 341                        | 177      |
| 4096          | 579                        | 704      |

进一步实现了local memory的kernel，代码改动较多，如下所示：

```opencl
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
```

使用该kernel的实验结果如下，分别测试5种不同大小的图片尺寸，local item size设置为8，即每个维度的工作组大小为8，每个工作组总共同时计算8*8个像素点。

| Picture width | OpenCL - Local memory (ms) | CPU (ms) |
| ------------- | -------------------------- | -------- |
| 256           | 263                        | 2        |
| 512           | 266                        | 11       |
| 1024          | 271                        | 44       |
| 2048          | 323                        | 177      |
| 4096          | 572                        | 704      |

### 4 实验结论

- 考虑针对特定卷积核的优化
- OpenCL的内存模型还不够熟练，比如哪些数据应该放在哪里我还不太懂
- 数据划分
- OpenCL只能传递一维数组