#include "../include/GPU_Arithmetics.h"
#include "../include/Timer.h"
#include "../include/CudaKernels.h"

#include <chrono>
#include <iostream>
#include <algorithm>


GPU_Arithmetics::GPU_Arithmetics()
{
    cudaFree(0);
}

SMP_Float GPU_Arithmetics::add(const SMP_Float &op1, const SMP_Float &op2)
{
    clearOperands();

    normalize(op1, op2);

    // Reverse Operands to fit in Radix Representation: x0*10e0 + x1*10e1 + ... + xn-1*10en-1
    std::reverse(op1Digits.begin(), op1Digits.end());
    std::reverse(op2Digits.begin(), op2Digits.end());

    Timer timer;
    timer.start();

    uint64_t size_op1 = op1Digits.size();
    uint64_t size_op2 = op2Digits.size();
    uint64_t sizeRes = op2Digits.size() + 1;

    // Create Host Data in CUDA-able representation, std::vector not possible
    const uint8_t *host_op1 = op1Digits.data();
    const uint8_t *host_op2 = op2Digits.data();

    // Define allocate and copy Device Data
    uint8_t *device_op1;
    uint8_t *device_op2;
    uint8_t *device_result;
    uint8_t *device_carry;

    cudaMalloc(&device_op1, size_op1 * sizeof(uint8_t));
    cudaMalloc(&device_op2, size_op2 * sizeof(uint8_t));
    cudaMalloc(&device_result, (sizeRes) * sizeof(uint8_t));
    cudaMalloc(&device_carry, (sizeRes) * sizeof(uint8_t));

    cudaMemcpy(device_op1, host_op1, size_op1 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_op2, host_op2, size_op2 * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Get the amount of blocks needed to perform addition of aArray with size "size"
    // Divide the size by amount of possible threads per block to perform addition per block
    int threadsPerBlock = 1024;
    int64_t blocks = (sizeRes / threadsPerBlock) + 1;

    uint8_t *host_result = new uint8_t[sizeRes];
    uint8_t *host_carry = new uint8_t[sizeRes];

    
    // Invoke kernel
    // Performs addition of two parameters without carry propagation

    
    addWithoutCarry<<<blocks, threadsPerBlock>>>(device_op1, device_op2, device_result, sizeRes - 1);

    do
    {
        // Note possible carries
        noteCarry<<<blocks, threadsPerBlock>>>(device_result, device_carry, sizeRes);
        // Add possible carries
        addCarry<<<blocks, threadsPerBlock>>>(device_result, device_carry, sizeRes);
        cudaMemcpy(host_carry, device_carry, (sizeRes) * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    } while (checkForCarry(host_carry, sizeRes)); // repeat as longs as there are carries left

    // copy result back to host
    cudaMemcpy(host_result, device_result, (sizeRes) * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    std::cout << "Duration GPU ADD \u001b[35m" << ": \u001b[0m" << timer.stop() << " µs" << std::endl;

    std::vector<uint8_t> toSMP;
    for (uint64_t i = 0; i < sizeRes; i++)
    {
        toSMP.push_back(host_result[i]);
    }

    delete[] host_carry;
    delete[] host_result;

    cudaFree(device_op1);
    cudaFree(device_op2);
    cudaFree(device_carry);
    cudaFree(device_result);

    return SMP_Float(toSMP, op1.exponent, op2.exponent, false);
}

SMP_Float GPU_Arithmetics::multiply(const SMP_Float &op1, const SMP_Float &op2)
{
    clearOperands();

    op1Digits = op1.digits;
    op2Digits = op2.digits;

    // Reverse Operands to fit in Radix Representation: x0*10e0 + x1*10e1 + ... + xn-1*10en-1
    std::reverse(op1Digits.begin(), op1Digits.end());
    std::reverse(op2Digits.begin(), op2Digits.end());

    Timer timer;
    timer.start();

    uint64_t size_op1 = op1Digits.size();
    uint64_t size_op2 = op2Digits.size();
    uint64_t sizeRes = op1.digits.size() + op2.digits.size();

    uint8_t *device_op1;
    uint8_t *device_op2;
    uint64_t *device_result;
    uint64_t *device_carry;

    cudaMalloc(&device_op1, size_op1 * sizeof(uint8_t));
    cudaMalloc(&device_op2, size_op2 * sizeof(uint8_t));
    cudaMalloc(&device_result, (sizeRes) * sizeof(uint64_t));
    cudaMalloc(&device_carry, (sizeRes+1) * sizeof(uint64_t));

    cudaMemcpy(device_op1, op1Digits.data(), size_op1 * sizeof(uint8_t), cudaMemcpyHostToDevice); // inline host op1
    cudaMemcpy(device_op2, op2Digits.data(), size_op2 * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Get the amount of blocks needed to perform addition of aArray with size "size"
    // Divide the size by amount of possible threads per block to perform addition per block
    int threadsPerBlock = 1024;
    int64_t blocks = (sizeRes / threadsPerBlock) + 1;

    // Performs addition of two parameters without carry propagation
    uint64_t *host_result = new uint64_t[sizeRes];
    uint64_t *host_carry = new uint64_t[sizeRes+1];

    for (uint64_t j = 0; j < size_op1; j++)
    {
        mulWithoutCarry<<<blocks, threadsPerBlock>>>(device_op1, device_op2, device_result, j, size_op2);
    }

    do
    {
        noteCarry<<<blocks, 1024>>>(device_result, device_carry, sizeRes);
        addCarry<<<blocks, 1024>>>(device_result, device_carry, sizeRes);
        cudaMemcpy(host_carry, device_carry, (sizeRes) * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    } while (checkForCarry(host_carry, sizeRes)); // repeat as longs as there are carries left

    // Copy result back to host
    cudaMemcpy(host_result, device_result, (sizeRes) * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    std::cout << "Duration GPU MUL \u001b[35m" << ": \u001b[0m" << timer.stop() << " µs" << std::endl;

    std::vector<uint8_t> toSMP;
    for (uint64_t i = 0; i < sizeRes; i++)
    {
        toSMP.push_back(static_cast<uint8_t>(host_result[i]));
    }

    delete[] host_carry;
    delete[] host_result;

    cudaFree(device_op1);
    cudaFree(device_op2);
    cudaFree(device_carry);
    cudaFree(device_result);
    
    return SMP_Float(toSMP, op1.exponent, op2.exponent, true);
}

// Checks if there occurred any carries while computing
bool GPU_Arithmetics::checkForCarry(uint8_t *carry, uint64_t size)
{
    for (int i = 0; i < size; i++)
    {
        if (carry[i] > 0)
        {
            return true;
        }
    }
    return false;
}

bool GPU_Arithmetics::checkForCarry(uint64_t *carry, uint64_t size)
{
    for (int i = 0; i < size; i++)
    {
        if (carry[i] > 0)
        {
            return true;
        }
    }
    return false;
}