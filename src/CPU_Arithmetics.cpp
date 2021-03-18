#include "../include/CPU_Arithmetics.h"
#include "../include/Timer.h"

#include <array>
#include <chrono>
#include <iostream>
#include <algorithm>

SMP_Float CPU_Arithmetics::add(const SMP_Float &op1, const SMP_Float &op2)
{
    clearOperands();

    normalize(op1, op2);

    // Reverse Operands to fit in Radix Representation: x0*10e0 + x1*10e1 + ... + xn-1*10en-1
    std::reverse(op1Digits.begin(), op1Digits.end());
    std::reverse(op2Digits.begin(), op2Digits.end());

    Timer timer;
    timer.start();

    uint8_t c = 0;
    uint8_t temp = 0;
    
    for (int i = 0; i < op1Digits.size(); i++)
    {
        temp = op1Digits[i] + op2Digits[i] + c;
        c = temp / 10;
        result.push_back(temp % 10);
    }
    result.push_back(c);
    std::cout << "Duration CPU ADD \u001b[35m" << ": \u001b[0m" << timer.stop() << " µs" << std::endl;

    return SMP_Float(result, op1.exponent, op2.exponent, false);
}

SMP_Float CPU_Arithmetics::multiply(const SMP_Float &op1, const SMP_Float &op2)
{
    clearOperands();

    op1Digits = op1.digits;
    op2Digits = op2.digits;

    // Reverse Operands to fit in Radix Representation: x0*10e0 + x1*10e1 + ... + xn-1*10en-1
    std::reverse(op1Digits.begin(), op1Digits.end());
    std::reverse(op2Digits.begin(), op2Digits.end());

    Timer timer;
    timer.start();

    // Calculate... a*b
    uint8_t c = 0;
    uint8_t temp = 0;
    uint64_t size = op1Digits.size() + op2Digits.size();

    std::vector<uint64_t> carry(size + 1, 0);
    std::vector<uint64_t> result_mul;

    for (int i = 0; i < size; i++)
    {
        result_mul.push_back(0);
        result.push_back(0);
    }

    for (uint64_t i = 0; i < op1Digits.size(); i++)
    {
        for (uint64_t k = 0; k < op2Digits.size(); k++)
        {
            result_mul[i + k] += op1Digits[i] * op2Digits[k];
        }
    }

    do
    {
        for (uint64_t i = 0; i < size; i++)
        {
            if (result_mul[i] >= 10)
            {
                carry[i + 1] = result_mul[i] / 10; 
                result_mul[i] = result_mul[i] % 10;
            }
            else
            {
                carry[i + 1] = 0;
            }
        }
        
        for (uint64_t i = 0; i < size; i++)
        {
            result_mul[i] += carry[i];
        }

    } while (checkForCarry(carry, size)); 

    std::cout << "Duration CPU MUL \u001b[35m" << ": \u001b[0m" << timer.stop() << " µs" << std::endl;

    for (uint64_t i = 0; i < size; i++)
    {
        result[i] = static_cast<uint8_t>(result_mul[i]);
    }

    return SMP_Float(result, op1.exponent, op2.exponent, true);
}

bool CPU_Arithmetics::checkForCarry(std::vector<uint64_t> &carry, uint64_t size)
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