#include "../include/IArithmetics.h"

void IArithmetics::normalize(const SMP_Float &op1, const SMP_Float &op2)
{
    uint64_t prec_op1 = op1.exponent >= 0 ? op1.digits.size() : std::abs(op1.exponent) + op1.digits.size();
    uint64_t prec_op2 = op2.exponent >= 0 ? op2.digits.size() : std::abs(op2.exponent) + op2.digits.size();

    uint64_t prec_res = prec_op1 + prec_op2;

    op1Digits = op1.digits;
    op2Digits = op2.digits;

    if (op1.exponent < op2.exponent)
    {
        op1Digits.insert(op1Digits.begin(), getAmountToShift(op2.exponent, op1.exponent), 0);
    }
    else if (op1.exponent > op2.exponent)
    {
        op2Digits.insert(op2Digits.begin(), getAmountToShift(op1.exponent, op2.exponent), 0);
    }

    op1Digits.insert(op1Digits.end(), prec_res - op1Digits.size(), 0);
    op2Digits.insert(op2Digits.end(), prec_res - op2Digits.size(), 0);
    uint64_t i = prec_res - 1;
    while((op1Digits[i] != 0) && (op2Digits[i] != 0))
    {
        op1Digits.erase(op1Digits.end());
        op2Digits.erase(op2Digits.end());
        i--;
    }
}

int64_t IArithmetics::getAmountToShift(int64_t biggerExponent, int64_t smallerExponent)
{
    int64_t amountToShift = smallerExponent - biggerExponent;
    return std::abs(amountToShift);
}

void IArithmetics::clearOperands()
{
    result.clear();
    op1Digits.clear();
    op2Digits.clear();
}