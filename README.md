# Sandra's Multiprecision Library 

## About
This library provides multiple precision arithmetics for floating point numbers accelerated with the GPU via CUDA (NVIDIA GPU required). The Code is written in C++ and CUDA C++.   
It is possible to add and to multiply two floating point operands with arbitrary precision.

## Features
 - Create multiple precision floating point numbers
 - Addition and multiplication of multiple precision floating point numbers on CPU and also with GPU acceleration via CUDA
 - Test class to execute performance tests

## Building from source
To build the downloaded project you need to have the latest CUDA Toolkit installed. Follow this Installation [Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

Afterwards build the project by starting the `build.sh` script
```
  $ ./build.sh
```
Now run the programm to perform some tests by starting the created executable
```
  $ ./test
```
## To-Do
- Improve of CUDA usage
- Optimize arithmetic algorithms, especially multiplication
- Reduce general overhead 

## References
 - [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
 - [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

 