#include "sharp.hpp"

const int mem_num = 3;
const int width = 2048;
const int local_work_item_size = 8;
// const int fil_size = 3;  // filter size

const int total_num = width * width;
const int overlap = fil_size - 1;
const int out_width = width - overlap;
const int total_num_out = out_width * out_width;

const int global_work_item_size = width;

int img_in[total_num] = {0};
int img_out[total_num_out] = {0};
int cpu_out[total_num_out] = {0};

void print_basic_info() {
  printf("=====basic info=====\n");
  printf("width: %d\n", width);
  printf("total_num: %d\n", total_num);
  printf("total_num_out: %d\n", total_num_out);
  printf("global_work_item_size: %d\n", global_work_item_size);
  printf("local_work_item_size: %d\n", local_work_item_size);
}

cl_context CreateContext() {
  cl_int errNum;
  cl_uint numPlatforms;
  cl_platform_id firstPlatformId;
  cl_context context = NULL;

  errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
  if (errNum != CL_SUCCESS || numPlatforms <= 0) {
    std::cout << "Failed to find any OpenCL platforms." << std::endl;
    return NULL;
  }

  cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformId, 0};
  context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);

  return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device) {
  cl_int errNum;
  cl_device_id *devices;
  cl_command_queue commandQueue = NULL;
  size_t deviceBufferSize = -1;

  errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);

  if (deviceBufferSize <= 0) {
    std::cout << "No devices available.";
    return NULL;
  }

  devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
  errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);

  commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

  *device = devices[0];
  delete[] devices;
  return commandQueue;
}

cl_program CreateProgram(cl_context context, cl_device_id device, const char *fileName) {
  cl_int errNum;
  cl_program program;

  std::ifstream kernelFile(fileName, std::ios::in);
  if (!kernelFile.is_open()) {
    std::cout << "Failed to open file for reading: " << fileName << std::endl;
    return NULL;
  }

  std::ostringstream oss;
  oss << kernelFile.rdbuf();

  std::string srcStdStr = oss.str();
  const char *srcStr = srcStdStr.c_str();
  program = clCreateProgramWithSource(context, 1, (const char **)&srcStr, NULL, NULL);

  errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  return program;
}

void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel1, cl_mem memObjects[mem_num]) {
  for (int i = 0; i < mem_num; i++)
    if (memObjects[i] != 0) clReleaseMemObject(memObjects[i]);

  if (commandQueue != 0) clReleaseCommandQueue(commandQueue);

  if (kernel1 != 0) clReleaseKernel(kernel1);

  if (program != 0) clReleaseProgram(program);

  if (context != 0) clReleaseContext(context);
}

int main(int argc, char **argv) {
  srand((unsigned)1107);

  print_basic_info();

  auto start_gpu = std::chrono::steady_clock::now();

  cl_context context = 0;
  cl_command_queue commandQueue = 0;
  cl_program program = 0;
  cl_device_id device = 0;
  cl_kernel kernel_conv = 0;
  cl_mem memObjects[mem_num] = {0, 0, 0};
  cl_int errNum;

  const char *filename = "sharp.cl";
  context = CreateContext();
  commandQueue = CreateCommandQueue(context, &device);
  program = CreateProgram(context, device, filename);
  kernel_conv = clCreateKernel(program, "conv", NULL);

  int filter[fil_size * fil_size] = {1, 1, 1, 1, -9, 1, 1, 1, 1};

  for (int i = 0; i < total_num; i++) {
    img_in[i] = rand() % 256;
    if (i < total_num_out) img_out[i] = -1;
  }

  // print_array(img_in, total_num, "img_in");
  cl_mem cl_img_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * total_num, img_in, NULL);
  cl_mem cl_img_out = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * total_num_out, NULL, NULL);
  cl_mem cl_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * fil_size * fil_size, filter, NULL);

  memObjects[0] = cl_img_in;
  memObjects[1] = cl_img_out;
  memObjects[2] = cl_filter;

  printf("mem inited\n");

  // ================================================================================================
  // STEP 1: calculate
  // ================================================================================================
  errNum = clSetKernelArg(kernel_conv, 0, sizeof(cl_mem), &cl_img_in);        // img_in
  errNum |= clSetKernelArg(kernel_conv, 1, sizeof(cl_mem), &cl_img_out);      // img_out
  errNum |= clSetKernelArg(kernel_conv, 2, sizeof(int), (void *)&out_width);  // img_out width
  errNum |= clSetKernelArg(kernel_conv, 3, sizeof(cl_mem), &cl_filter);       // filter kernel
  errNum |= clSetKernelArg(kernel_conv, 4, sizeof(int), (void *)&fil_size);   // filter size
  int local_mem_size = (local_work_item_size + overlap) * (local_work_item_size + overlap);
  errNum |= clSetKernelArg(kernel_conv, 5, sizeof(int) * local_mem_size, 0);   // filter size

  if (errNum != CL_SUCCESS) {
    printf("set arg error\n");
    std::cerr << getErrorString(errNum) << std::endl;
    Cleanup(context, commandQueue, program, kernel_conv, memObjects);
    return errNum;
  } else {
    printf("set arg success\n");
  }

  cl_uint work_dim = 2;
  size_t globalWorkSize[2] = {global_work_item_size, global_work_item_size};
  size_t localWorkSize[2] = {local_work_item_size, local_work_item_size};

  // init cl memory
  errNum = clEnqueueWriteBuffer(commandQueue, memObjects[0], CL_TRUE, 0, sizeof(int) * total_num, img_in, 0, NULL, NULL);
  if (errNum != CL_SUCCESS) {
    printf("init memory error\n");
    std::cerr << getErrorString(errNum) << std::endl;
    Cleanup(context, commandQueue, program, kernel_conv, memObjects);
    return errNum;
  } else {
    printf("init memory success\n");
  }

  // execute kernel: compute histogram
  printf("global_item_size: %d, local_item_size: %d\n", global_work_item_size, local_work_item_size);
  errNum = clEnqueueNDRangeKernel(commandQueue, kernel_conv, work_dim, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  if (errNum != CL_SUCCESS) {
    printf("exe error\n");
    std::cerr << getErrorString(errNum) << std::endl;
    Cleanup(context, commandQueue, program, kernel_conv, memObjects);
    return errNum;
  } else {
    printf("kernel_conv done\n");
  }

  // read cl memory
  errNum = clEnqueueReadBuffer(commandQueue, memObjects[1], CL_TRUE, 0, sizeof(int) * total_num_out, img_out, 0, NULL, NULL);
  // print_array(img_out, total_num_out, "img_out");
  Cleanup(context, commandQueue, program, kernel_conv, memObjects);

  std::cout << "gpu(ms)=" << since(start_gpu).count() << std::endl;

  auto start_cpu = std::chrono::steady_clock::now();
  // STEP 2: compare with cpu result
  cpu_sharp(img_in, cpu_out, width, out_width);
  // print_array(cpu_out, total_num_out, "cpu_out");

  if (compare(cpu_out, img_out, total_num_out)) {
    printf("\ncompare with cpu result success\n");
  } else {
    printf("\ncompare with cpu result failed\n");
  }
  std::cout << "cpu(ms)=" << since(start_cpu).count() << std::endl;

  return 0;
}
