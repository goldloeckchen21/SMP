#!/bin/bash
nvcc main.cpp -o test -link src/SMP_Float.cpp -link src/CPU_Arithmetics.cpp -link src/GPU_Arithmetics.cu -link src/SMP_Test.cpp -link src/Timer.cpp -link src/IArithmetics.cpp