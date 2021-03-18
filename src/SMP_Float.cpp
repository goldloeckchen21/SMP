#include "../include/SMP_Float.h"

#include <iostream>
#include <algorithm>

SMP_Float::SMP_Float(std::string input)
{
    normalizeMantissa(input);
    convertMantissaToVector();
}

SMP_Float::SMP_Float(const std::vector<uint8_t> &resultDigits, int64_t exp1, int64_t exp2, bool multiplication)
{
    digits = resultDigits;
    uint64_t normedExponent = 0;
    if (!multiplication)
    {
        if (digits[digits.size() - 1] > 0)
        {
            normedExponent++;
        }
    }

    while (digits[0] == 0)
    {
        digits.erase(digits.begin());
    }

    std::reverse(digits.begin(), digits.end());

    if (multiplication)
    {
        normedExponent = 0;
        while (digits[0] == 0)
        {
            digits.erase(digits.begin());
            normedExponent--;
        }
    }
    else
    {
        normedExponent += exp1 > exp2 ? exp1 : exp2;
        while (digits[0] == 0)
        {
            digits.erase(digits.begin());
        }
    }

    std::string resultMantissa;

    for (uint64_t i = 0; i < digits.size(); i++)
    {
        resultMantissa.push_back(digits[i] + 48);
    }

    resultMantissa.insert(resultMantissa.begin(), '.');
    resultMantissa.insert(resultMantissa.begin(), '0');

    int64_t sumExponentOperands = exp1 + exp2;
    normalizeMantissa(resultMantissa);

    if (multiplication)
    {
        exponent = normedExponent + exp1 + exp2;
    }
    else
    {
        exponent = normedExponent;
    }
}

void SMP_Float::normalizeMantissa(std::string toNorm)
{
    bool startsWithZero = toNorm.front() == '0';
    auto predecimal = toNorm.find('.');

    uint64_t leadingZeros = 0;
    uint64_t startIndex = 0;

    if (startsWithZero)
    {
        leadingZeros = getLeadingZeros(toNorm.substr(2));
    }

    if (startsWithZero)
    {
        exponent = 0;
        startIndex = 2;
        if (leadingZeros > 0)
        {
            exponent = leadingZeros * (-1);
            startIndex += leadingZeros;
        }
        mantissa = toNorm.substr(startIndex, toNorm.size());
    }
    else
    {
        exponent = predecimal;
        int zerosToDeleteAfterComma = 0;
        if ((toNorm.size() == exponent + 2) && (toNorm[exponent + 1] == '0'))
        {
            mantissa = toNorm.erase(toNorm.find('.'), 2);
        }
        else
        {
            mantissa = toNorm.erase(toNorm.find('.'), 1);
        }
    }
}

uint64_t SMP_Float::getLeadingZeros(std::string input)
{
    uint64_t i = 0;
    while (input[i] == '0' && i < input.size())
    {
        i++;
    }
    return i;
}

void SMP_Float::convertMantissaToVector()
{
    const auto asciiOffset = 48;
    for (const char character : mantissa)
    {
        digits.push_back(character - asciiOffset);
    }
}

void SMP_Float::printStatus()
{
    std::cout << "****** Status of SMP_Float: ******" << std::endl;
    std::cout << "Normalized mantissa: " << mantissa << std::endl;
    std::cout << "Length of mantissa: " << mantissa.size() << std::endl;
    std::cout << "Exponent: " << exponent << std::endl;
    std::cout << "Digits Size: " << digits.size() << std::endl;
    std::cout << "Digits: ";
    printDigits();
    std::cout << std::endl;
    std::cout << "Float: "
              << "0." << mantissa << "x10e" << exponent << std::endl;
    std::cout << "----------------------------------" << std::endl;
}

void SMP_Float::printDigits()
{
    for (const auto digit : digits)
    {
        std::cout << unsigned(digit);
    }
}