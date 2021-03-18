#ifndef CUDAKERNELS_H
#define CUDAKERNELS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

/**
 * Performs Addition on GPU of two Parameters without Carry
 * Parameters:
 *          op1: Operand 1 Array
 *          op2: Operand 2 Array
 *          res: Result Array 
 *          size: size of the expected Result
 */
__global__ void addWithoutCarry(uint8_t *op1, uint8_t *op2, uint8_t *res, uint64_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < size)
    {
        res[index] = op1[index] + op2[index];
    }
}

// Overloaded Carry Handling for Multiplication
/**
 * Saves possible Carry on GPU while Performing Multiplication and adjusts Result Array
 * Parameter:
 *          res: Result Array with temporay results
 *          carry: Carry Array
 *          size: size of the expected Result
 */
__global__ void noteCarry(uint64_t *res, uint64_t *carry, uint64_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        if (res[index] >= 10)
        {
            carry[index + 1] = res[index] / 10;
            res[index] = res[index] % 10;
        }
        else
        {
            carry[index + 1] = 0;
        }
    }
}

/**
 * Adds the possible Carry on GPU to the Result
 * Parameter:
 *          res: Result Array with temporay results
 *          carry: Carry Array
 *          size: size of the expected Result
 */
__global__ void addCarry(uint64_t *res, uint64_t *carry, uint64_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        res[index] += carry[index];
    }
}

/**
 * Performs Multiplication on GPU of two Parameters without Carry
 * Parameters:
 *          op1: Operand 1 Array
 *          op2: Operand 2 Array
 *          res: Result Array 
 *          size: size of the expected Result
 */
__global__ void mulWithoutCarry(uint8_t *op1, uint8_t *op2, uint64_t *res, uint64_t iterator, uint64_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        res[iterator + index] += op1[iterator] * op2[index];
    }
}

/**
 * Saves possible Carry on GPU while Performing Addition and adjusts Result Array
 * Parameter:
 *          res: Result Array with temporay results
 *          carry: Carry Array
 *          size: size of the expected Result
 */
__global__ void noteCarry(uint8_t *res, uint8_t *carry, uint64_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        if (res[index] >= 10)
        {
            carry[index + 1] = res[index] / 10;
            res[index] = res[index] % 10;
        }
        else
        {
            carry[index + 1] = 0;
        }
    }
}

/**
 * Adds the possible Carry on GPU to the Result
 * Parameter:
 *          res: Result Array with temporay results
 *          carry: Carry Array
 *          size: size of the expected Result
 */
__global__ void addCarry(uint8_t *res, uint8_t *carry, uint64_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        res[index] += carry[index];
    }
}

#endif