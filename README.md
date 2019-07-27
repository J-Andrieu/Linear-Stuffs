# Linear Math n' Stuff

## Description
This doesn't actually have a name yet, it's just a simple little namespace for making linear algebra easier to do in c++. The main.cpp file is just a test driver to make sure everything is running properly.

## Dependency Instructions
This project uses OpenCL, so just make sure your GPU drivers are up to date and you should be good to go.

## Building and Running - CMake Instructions
The building of the project is done using CMake.

Quick Usage with defaults
```bash
mkdir build
cd build
cmake ..
make
./matrix_test
```

If you're using windows and it's refusing to locate the kernels directory
```bash
mkdir build
cd build
cmake .. -DMATRIX_KERNEL_DIR="Absolute/path/to/kernels"
make
./matrix_test
```
