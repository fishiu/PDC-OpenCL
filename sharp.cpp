#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "sharp.hpp"

const int mem_num = 3;

const int width = 16;
const int total_num = width * width;
// const int fil_size = 3;  // filter size
const int overlap = fil_size - 1;
const int out_width = width - overlap;
const int total_num_out = out_width * out_width;
const int item_size = 4;  // how many data point a work item need to handle
const int global_work_item_size = ceil((float)out_width / item_size);  // 2D
const int local_work_item_size = 1;

void print_basic_info() {
  printf("=====basic info=====\n");
  printf("width: %d\n", width);
  printf("total_num: %d\n", total_num);
  printf("item_size: %d\n", item_size);
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

bool CreateMemObjects(cl_context context, cl_mem memObjects[mem_num]) {
  // data_in
  memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * total_num, NULL, NULL);
  // data_out
  memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * total_num_out, NULL, NULL);
  // log
  memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * total_num, NULL, NULL);
  return true;
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

  cl_context context = 0;
  cl_command_queue commandQueue = 0;
  cl_program program = 0;
  cl_device_id device = 0;
  cl_kernel kernel_conv = 0;
  cl_mem memObjects[mem_num] = {0, 0};
  cl_int errNum;

  const char *filename = "sharp.cl";
  context = CreateContext();
  commandQueue = CreateCommandQueue(context, &device);
  program = CreateProgram(context, device, filename);
  kernel_conv = clCreateKernel(program, "conv", NULL);

  int img_in[total_num];
  int img_out[total_num];
  int log[total_num];
  // const int filter[fil_size][fil_size] = {1, 1, 1, 1, -9, 1, 1, 1, 1};

  for (int i = 0; i < total_num; i++) {
    img_in[i] = rand() % 256;
    log[i] = 0;
    if (i < total_num_out)
      img_out[i] = -1;
  }

  print_array(img_in, total_num, "img_in");

  if (!CreateMemObjects(context, memObjects)) {
    Cleanup(context, commandQueue, program, kernel_conv, memObjects);
    return 1;
  }

  // ================================================================================================
  // STEP 1: calculate
  // ================================================================================================
  errNum = clSetKernelArg(kernel_conv, 0, sizeof(cl_mem), &memObjects[0]);    // img_in
  if(errNum != CL_SUCCESS) {
    printf("set arg error\n");
    std::cerr << getErrorString(errNum) << std::endl;
    return errNum;
  }
  errNum |= clSetKernelArg(kernel_conv, 1, sizeof(cl_mem), &memObjects[1]);   // img_out
  errNum |= clSetKernelArg(kernel_conv, 2, sizeof(int), (void *)&width);      // img width
  errNum |= clSetKernelArg(kernel_conv, 3, sizeof(int), (void *)&out_width);  // img_out width
  errNum |= clSetKernelArg(kernel_conv, 4, sizeof(int), (void *)&item_size);  // item_size
  // errNum |= clSetKernelArg(kernel_conv, 5, sizeof(int), (void *)filter);     // filter kernel
  errNum |= clSetKernelArg(kernel_conv, 5, sizeof(cl_mem), &memObjects[2]);     // filter kernel

  if(errNum != CL_SUCCESS) {
    printf("set arg error\n");
    std::cerr << getErrorString(errNum) << std::endl;
    return errNum;
  }

  cl_uint work_dim = 2;
  size_t globalWorkSize[2] = {global_work_item_size, global_work_item_size};
  size_t localWorkSize[2] = {local_work_item_size, local_work_item_size};

  // init cl memory
  errNum = clEnqueueWriteBuffer(commandQueue, memObjects[0], CL_TRUE, 0, sizeof(int) * total_num, img_in, 0, NULL, NULL);
  if(errNum != CL_SUCCESS) {
    printf("init memory error\n");
    std::cerr << getErrorString(errNum) << std::endl;
    return errNum;
  }
  errNum = clEnqueueWriteBuffer(commandQueue, memObjects[2], CL_TRUE, 0, sizeof(int) * total_num, log, 0, NULL, NULL);
  if(errNum != CL_SUCCESS) {
    printf("init memory error\n");
    std::cerr << getErrorString(errNum) << std::endl;
    return errNum;
  }
  
  // execute kernel: compute histogram
  printf("global_item_size: %d, local_item_size: %d\n", global_work_item_size, local_work_item_size);
  errNum = clEnqueueNDRangeKernel(commandQueue, kernel_conv, work_dim, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  if(errNum != CL_SUCCESS) {
    printf("exe error\n");
    std::cerr << getErrorString(errNum) << std::endl;
    return errNum;
  }
  printf("kernel_conv done\n");

  // read cl memory
  errNum = clEnqueueReadBuffer(commandQueue, memObjects[1], CL_TRUE, 0, sizeof(int) * total_num_out, img_out, 0, NULL, NULL);
  if(errNum != CL_SUCCESS) {
    std::cerr << getErrorString(errNum) << std::endl;
    return errNum;
  }
  errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, sizeof(int) * total_num, log, 0, NULL, NULL);
  if(errNum != CL_SUCCESS) {
    std::cerr << getErrorString(errNum) << std::endl;
    return errNum;
  }

  print_array(log, total_num, "log");
  print_array(img_out, total_num_out, "img_out");

  // STEP 2: compare with cpu result
  int cpu_out[total_num_out] = {0};
  cpu_sharp(img_in, cpu_out, width, out_width);
  print_array(cpu_out, total_num_out, "cpu_out");

  if(compare(cpu_out, img_out, total_num_out)) {
    printf("\ncompare with cpu result success\n");
  } else {
    printf("\ncompare with cpu result failed\n");
  }

  Cleanup(context, commandQueue, program, kernel_conv, memObjects);
  return 0;
}
