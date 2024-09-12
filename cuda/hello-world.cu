#include <stdio.h>
// 核函数。
__global__ void add() {
  auto blockId = gridDim.x * blockIdx.y + blockIdx.x;
  auto tid = blockId * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
  printf("block id: %d, thread id: %d\n", blockId, tid);
}
int main() {

  dim3 grid_size(2,3);
  dim3 block_size(4,4);
  add<<<grid_size,block_size>>>();
  cudaDeviceSynchronize();
  return 0;
}


