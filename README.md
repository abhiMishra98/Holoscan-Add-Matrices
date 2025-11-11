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
```bash
cmake ../src -DCMAKE_PREFIX_PATH=/opt/nvidia/holoscan
```

3. Build the project:
```bash
make -j
```

## Profiling using Nsight systems

Once, makefile is generated run the following code
```bash
nsys profile   --trace=cuda,nvtx,osrt   --gpu-metrics-devices=all   --sample=cpu   --output=profile_report3   ./cudaTest
```

Open Nsight systems UI and locate the file
```bash
nsys-ui
```
## Profiling using Nsight compute

Once, makefile is generated run the following code
```bash
ncu --replay-mode application -o compute_report ./cudaTest
```

Open Nsight systems UI and locate the file
```bash
ncu-ui compute_report.ncu-rep
```
