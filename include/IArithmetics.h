#ifndef IARITHMETICS_H
#define IARITHMETICS_H

#include "SMP_Float.h"

class IArithmetics
{
public:
    virtual SMP_Float add(const SMP_Float &op1, const SMP_Float &op2) = 0;
    virtual SMP_Float multiply(const SMP_Float &op1, const SMP_Float &op2) = 0;
    virtual ~IArithmetics() = default; // Default destructor to prevent compile warnings

protected:
    void normalize(const SMP_Float &op1, const SMP_Float &op2);
    int64_t getAmountToShift(int64_t biggerExponent, int64_t smallerExponent);
    void clearOperands();

    std::vector<uint8_t> result;
    std::vector<uint8_t> op1Digits;
    std::vector<uint8_t> op2Digits;
};

#endif