#include <math.h>
// #include <stdlib.h>
#include <stdio.h>

const double EPSILON = 1.0e-10;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

// 核函数。
__global__ void add1Dim(const double *x, const double *y, double *z,
                        const int N);
__global__ void add2Dim(const double *x, const double *y, double *z,
                        const int N);
__global__ void add3Dim(const double *x, const double *y, double *z,
                        const int N);
// 重载设备函数。
__device__ double add_in_device(const double x, const double y);
__device__ void add_in_device(const double x, const double y, double &z);

// 主机函数。
void check(const double *z, const int N);

int main() {
  const int N = 1e4;
  const int M = sizeof(double) * N;

  // 申请主机内存。
  // 支持使用 new-delete 方式创建和释放内存。
  // double *h_x = (double*) malloc(M);
  double *h_x = new double[N];
  double *h_y = (double *)malloc(M);
  double *h_z = (double *)malloc(M);

  // 初始化主机数据。
  for (int i = 0; i < N; ++i) {
    h_x[i] = a;
    h_y[i] = b;
  }

  // 申请设备内存。
  // cudeError_t cudaMalloc(void **address, size_t size);
  double *d_x, *d_y, *d_z;
  cudaMalloc((void **)&d_x, M);
  cudaMalloc((void **)&d_y, M);
  cudaMalloc((void **)&d_z, M);

  // 从主机复制数据到设备。
  // cudaError_t cudaMemcpy(void *dst, void *src, size_t count, enum
  // cudaMemcpyKind kind); kind 可以简化使用
  // `cudaMemcpyDefault`，由系统自动判断拷贝方向（x64主机）。
  cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

  dim3 block_size1(100);
  dim3 grid_size1(101); // 线程数应该不少于计算数目。
  // 在设备中执行计算。
  dim3 block_size2(32, 4);
  dim3 grid_size2(N / 256 + 1, 2); // 线程数应该不少于计算数目。
  dim3 block_size3(25, 2, 2);
  dim3 grid_size3(26, 2, 2); // 线程数应该不少于计算数目。
  // add1Dim<<<grid_size1, block_size1>>>(d_x, d_y, d_z, N);
  // add2Dim<<<grid_size1, block_size1>>>(d_x, d_y, d_z, N);
  add3Dim<<<grid_size3, block_size3>>>(d_x, d_y, d_z, N);

  // 从设备复制数据到主机。
  cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
  check(h_z, N);

  // 释放主机内存。
  // free(h_x);
  if (h_x)
    delete[] h_x;
  free(h_y);
  free(h_z);

  // 释放设备内存。
  // cudaError_t cudaFree(void *address);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  return 0;
}

__global__ void add1Dim(const double *x, const double *y, double *z,
                        const int N) {
  // 在主机函数中需要依次对每个元素进行操作，需要使用一个循环。
  // 在设备函数中，因为采用“单指令-多线程”方式，所以可以去掉循环、只要将数组元素索引和线程索引一一对应即可。
  const int tid = blockIdx.x * (blockDim.x * blockDim.y) +
                  threadIdx.y * blockDim.x + threadIdx.x;
  if (tid > N)
    return;

  if (tid % 4 == 0) {
    z[tid] = add_in_device(x[tid], y[tid]);
  } else {
    add_in_device(x[tid], y[tid], z[tid]);
  }
}

__global__ void add2Dim(const double *x, const double *y, double *z,
                        const int N) {
  // 在主机函数中需要依次对每个元素进行操作，需要使用一个循环。
  // 在设备函数中，因为采用“单指令-多线程”方式，所以可以去掉循环、只要将数组元素索引和线程索引一一对应即可。
  auto blockId = blockIdx.x + blockIdx.y * gridDim.x;
  const int tid = (blockDim.x * blockDim.y) * blockId +
                  blockDim.x * threadIdx.y + threadIdx.x;
  if (tid > N)
    return;

  if (tid % 4 == 0) {
    z[tid] = add_in_device(x[tid], y[tid]);
  } else {
    add_in_device(x[tid], y[tid], z[tid]);
  }
}

__device__ double add_in_device(const double x, const double y) {
  return x + y;
}

__device__ void add_in_device(const double x, const double y, double &z) {
  z = x + y;
}

void check(const double *z, const int N) {
  bool has_error = false;
  for (int i = 0; i < N; ++i) {
    if (fabs(z[i] - c) > EPSILON) {
      // printf("%d, %f, %f\n", i, z[i], c);
      has_error = true;
    }
  }

  printf("cuda; %s\n", has_error ? "has error" : "no error");
}

__global__ void add3Dim(const double *x, const double *y, double *z,
                        const int N) {
  // 3 dimension block calculation
  auto blockId = (gridDim.x * gridDim.y) * blockIdx.z + gridDim.x * blockIdx.y +
                 blockIdx.x;
  const int tid = (blockDim.x * blockDim.y * blockDim.z) * blockId +
                  threadIdx.z * (blockDim.x * blockDim.y) +
                  blockDim.x * threadIdx.y + threadIdx.x;

  if (tid > N)
    return;

  if (tid % 4 == 0) {
    z[tid] = add_in_device(x[tid], y[tid]);
  } else {
    add_in_device(x[tid], y[tid], z[tid]);
  }
}
