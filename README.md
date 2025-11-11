# Holoscan Add Matrices

This project demonstrates a simple matrix addition application using **NVIDIA Holoscan** with CUDA.

---

## Build Instructions

1. Create a build folder and navigate inside:

```bash
mkdir build
cd build
```

2. Configure the project using CMake (make sure to set the Holoscan path):
cmake ../src -DCMAKE_PREFIX_PATH=/opt/nvidia/holoscan

3. Build the project:
make -j
