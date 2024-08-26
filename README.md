## Build llvm source code & triton

### check triton version

> here use release/3.0.x branch, llvm hash: 10dc3a8e916d73291269e5e2b82dd22681489aa1

### Build llvm source code

```shell
git checkout 10dc3a8e916d73291269e5e2b82dd22681489aa1
cmake -G Ninja -B build -S llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang" \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_ENABLE_LLD=ON
```
> LLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" can only be `X86,AMD,Nvidia`

### Build triton

```shell
git clone --recursive https://github.com/triton-lang/triton.git
cd triton
export LLVM_BUILD_DIR=/home/triton/work/llvm-source-code/build
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_SYSPATH=$LLVM_BUILD_DIR
export TRITON_BUILD_WITH_CCACHE=true
export PATH=/home/triton/work/llvm-source-code/build/bin:$PATH
pip install -e python
```
> Change `LLVM_BUILD_DIR` based on your build
