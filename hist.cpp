#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "hist.hpp"

const int hist_size = 256;
const int mem_num = 4;

const int width = 8;
const int height = 8;
const int total_num = width * height;
const int size_per_item = 4;
const int global_work_item_size = total_num / size_per_item;
const int local_work_item_size = 1;

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
  // log
  memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * total_num, NULL, NULL);
  // hist
  memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * hist_size, NULL, NULL);
  // hist eq
  memObjects[3] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * hist_size, NULL, NULL);
  return true;
}

void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem memObjects[3]) {
  for (int i = 0; i < mem_num; i++) {
    if (memObjects[i] != 0) clReleaseMemObject(memObjects[i]);
  }
  if (commandQueue != 0) clReleaseCommandQueue(commandQueue);

  if (kernel != 0) clReleaseKernel(kernel);

  if (program != 0) clReleaseProgram(program);

  if (context != 0) clReleaseContext(context);
}

int main(int argc, char **argv) {
  srand((unsigned)1107);

  cl_context context = 0;
  cl_command_queue commandQueue = 0;
  cl_program program = 0;
  cl_device_id device = 0;
  cl_kernel kernel_gethist = 0, kernel_histimg = 0;
  cl_mem memObjects[4] = {0, 0, 0, 0};
  cl_int errNum;

  const char *filename = "hist.cl";
  context = CreateContext();
  commandQueue = CreateCommandQueue(context, &device);
  program = CreateProgram(context, device, filename);
  kernel_gethist = clCreateKernel(program, "imgToHist", NULL);
  kernel_histimg = clCreateKernel(program, "histEqToImg", NULL);

  int img[total_num];
  int imgeq[total_num];
  // todo compare with cpu result
  int log[total_num];
  int hist[hist_size];
  int histeq[hist_size];

  for (int i = 0; i < total_num; i++) {
    img[i] = rand() % 256;
    imgeq[i] = 0;
    log[i] = 0;
  }

  for (int i = 0; i < hist_size; i++) {
    hist[i] = 0;
    histeq[i] = 0;
  }

  if (!CreateMemObjects(context, memObjects)) {
    Cleanup(context, commandQueue, program, kernel_gethist, memObjects);
    return 1;
  }

  // STEP 1: get histogram
  errNum = clSetKernelArg(kernel_gethist, 0, sizeof(cl_mem), &memObjects[0]);        // img
  errNum |= clSetKernelArg(kernel_gethist, 1, sizeof(cl_mem), &memObjects[2]);       // hist
  errNum |= clSetKernelArg(kernel_gethist, 2, sizeof(int) * hist_size, NULL);        // local
  errNum |= clSetKernelArg(kernel_gethist, 3, sizeof(int), (void *)&size_per_item);  // size_per_item
  errNum |= clSetKernelArg(kernel_gethist, 4, sizeof(cl_mem), &memObjects[1]);       // log

  size_t globalWorkSize[1] = {global_work_item_size};
  size_t localWorkSize[1] = {local_work_item_size};

  errNum = clEnqueueWriteBuffer(commandQueue, memObjects[0], CL_TRUE, 0, sizeof(int) * total_num, img, 0, NULL, NULL);
  errNum = clEnqueueWriteBuffer(commandQueue, memObjects[1], CL_TRUE, 0, sizeof(int) * total_num, log, 0, NULL, NULL);
  
  printf("global_item_size: %d, local_item_size: %d\n", global_work_item_size, local_work_item_size);
  errNum = clEnqueueNDRangeKernel(commandQueue, kernel_gethist, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

  errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, sizeof(int) * hist_size, hist, 0, NULL, NULL);
  errNum = clEnqueueReadBuffer(commandQueue, memObjects[1], CL_TRUE, 0, sizeof(int) * total_num, log, 0, NULL, NULL);

  printf("\nimg:\n");
  for (int i = 0; i < total_num; i++) {
    printf("%d\t", img[i]);
  }
  printf("\n");
  
  // printf("\nhist:\n");
  // for (int i = 0; i < hist_size; i++) {
  //   printf("%d\t", hist[i]);
  // }
  // printf("\n");


  // STEP 2: hist equalization
  printf("\nconvert hist to histeq\n");
  get_histeq(hist, histeq, total_num);
  printf("\n");
  
  // write histeq to mem
  for (int i = 0; i < total_num; i++) log[i] = 0;  // reset log
  errNum = clEnqueueWriteBuffer(commandQueue, memObjects[3], CL_TRUE, 0, sizeof(int) * hist_size, histeq, 0, NULL, NULL);
  errNum = clEnqueueWriteBuffer(commandQueue, memObjects[1], CL_TRUE, 0, sizeof(int) * total_num, log, 0, NULL, NULL);

  // set args to call hist_to_img
  errNum = clSetKernelArg(kernel_histimg, 0, sizeof(cl_mem), &memObjects[0]);        // img
  errNum |= clSetKernelArg(kernel_histimg, 1, sizeof(cl_mem), &memObjects[3]);       // histeq
  errNum |= clSetKernelArg(kernel_histimg, 2, sizeof(cl_mem), &memObjects[1]);       // histeq
  // execute kernel
  globalWorkSize[0] = total_num;
  localWorkSize[0] = 1;
  errNum = clEnqueueNDRangeKernel(commandQueue, kernel_histimg, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  // write mem_img to imgeq
  errNum = clEnqueueReadBuffer(commandQueue, memObjects[0], CL_TRUE, 0, sizeof(int) * total_num, imgeq, 0, NULL, NULL);
  errNum = clEnqueueReadBuffer(commandQueue, memObjects[1], CL_TRUE, 0, sizeof(int) * total_num, log, 0, NULL, NULL);


  printf("\nlog:\n");
  for (int i = 0; i < total_num; i++) {
    printf("%d\t", log[i]);
  }
  printf("\n");

  printf("\nimgeq:\n");
  for (int i = 0; i < total_num; i++) {
    printf("%d\t", imgeq[i]);
  }
  printf("\n");

  // STEP 3: compare with cpu result
  int cpu_imgeq[total_num] = {0};
  cpu_histeq(img, cpu_imgeq, total_num);

  printf("\ncpu_imgeq:\n");
  for (int i = 0; i < total_num; i++) {
    printf("%d\t", cpu_imgeq[i]);
  }
  printf("\n");

  if(compare(cpu_imgeq, imgeq, total_num)) {
    printf("\ncompare with cpu result success\n");
  } else {
    printf("\ncompare with cpu result failed\n");
  }

  Cleanup(context, commandQueue, program, kernel_gethist, memObjects);
  return 0;
}
