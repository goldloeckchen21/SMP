#include "../include/SMP_Test.h"
#include "../include/Timer.h"

#include <iostream>

int64_t SMP_Test::executeTests(IArithmetics *calculator, bool performanceTest)
{
    Timer timer;
    timer.start();

    testAdd(calculator);
    testMultiply(calculator);
    if (performanceTest)
    {
        testPerformance(calculator, false);
    }

    return timer.stop();
}

void SMP_Test::testAdd(IArithmetics *calculator)
{
    int count = 1;
    bool mantissa_success = false;
    bool exponent_success = false;

    for (values input : addTest)
    {
        SMP_Float op1 = SMP_Float(input.op1);
        SMP_Float op2 = SMP_Float(input.op2);
        SMP_Float res = calculator->add(op1, op2);

        std::string resExponent = std::to_string(res.exponent);

        std::cout << "----- Test \u001b[33mADDITION\u001b[0m [" << count << "]" /*with operands: " << input.op1 << " and " << input.op2 << */ "-----" << std::endl;

        if (res.mantissa.compare(input.res) == 0)
        {
            mantissa_success = true;
        }
        else
        {
            std::cout << "Mantissa Test " << count << ": NOT OK!" << std::endl;
            std::cout << "res Mantissa: " << res.mantissa << " != testMantissa: " << input.res << std::endl;
            std::cout << "size res Mantissa: " << res.mantissa.size() << " !=  size testMantissa: " << input.res.size() << std::endl;
            mantissa_success = false;
        }
        if (resExponent.compare(input.exp) == 0)
        {
            exponent_success = true;
        }
        else
        {
            std::cout << "Exponent Test " << count << ": NOT OK!" << std::endl;
            std::cout << "res Exponent: " << resExponent << " != test Exponent: " << input.exp << std::endl;
            std::cout << "size res Exp: " << resExponent.size() << " !=  size testExp: " << input.exp.size() << std::endl;
            exponent_success = false;
        }
        std::cout << std::endl;
        if (mantissa_success && exponent_success)
        {
            std::cout << "\u001b[32m--------------- TEST SUCCEDED-------------------\u001b[0m" << std::endl;
        }
        else
        {
            std::cout << "\u001b[31m------------- TEST NOT SUCCEDED ----------------\u001b[0m" << std::endl;
        }
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << std::endl;

        count++;
    }
}

void SMP_Test::testMultiply(IArithmetics *calculator)
{
    int count = 1;
    bool mantissa_success = false;
    bool exponent_success = false;

    for (values input : mulTest)
    {
        SMP_Float op1 = SMP_Float(input.op1);
        SMP_Float op2 = SMP_Float(input.op2);
        SMP_Float res = calculator->multiply(op1, op2);
        std::string resExponent = std::to_string(res.exponent);

        std::cout << "----- Test \u001b[36mMULTIPLICATION\u001b[0m [" << count << "] " /*with operands: " << input.op1 << " and " << input.op2 << */ "-----" << std::endl;
        if (res.mantissa.compare(input.res) == 0)
        {
            mantissa_success = true;
        }
        else
        {
            std::cout << "Mantissa Test " << count << ": NOT OK!" << std::endl;
            std::cout << "res Mantissa: " << res.mantissa << " != testMantissa: " << input.res << std::endl;
            std::cout << "size res Mantissa: " << res.mantissa.size() << " !=  size testMantissa: " << input.res.size() << std::endl;
            mantissa_success = false;
        }

        if (resExponent.compare(input.exp) == 0)
        {
            exponent_success = true;
        }
        else
        {
            std::cout << "Exponent Test " << count << ": NOT OK!" << std::endl;
            std::cout << "res Exponent: " << resExponent << " != test Exponent: " << input.exp << std::endl;
            std::cout << "size res Exp: " << resExponent.size() << " !=  size testExp: " << input.exp.size() << std::endl;
            exponent_success = false;
        }
        std::cout << std::endl;

        if (mantissa_success && exponent_success)
        {
            std::cout << "\u001b[32m--------------- TEST SUCCEDED-------------------\u001b[0m" << std::endl;
        }
        else
        {
            std::cout << "\u001b[31m------------- TEST NOT SUCCEDED ----------------\u001b[0m" << std::endl;
        }
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << std::endl;

        count++;
    }
}

void SMP_Test::testPerformance(IArithmetics *calculator, bool worstCase)
{
    for (int precision : perfTest)
    {
        char digit = worstCase ? '9' : '1';
        std::string bigFloat = createBigFloat(0, precision, digit);
        SMP_Float operand = SMP_Float(bigFloat);

        Timer timer;

        timer.start();
        calculator->add(operand, operand);
        std::cout << "Duration Worst Case ADD with precision \u001b[35m" << precision << ": \u001b[0m" << timer.stop() << " µs" << std::endl;

        timer.start();
        calculator->multiply(operand, operand);
        std::cout << "Duration Worst Case MUL with precision \u001b[35m" << precision << ": \u001b[0m" << timer.stop() << " µs" << std::endl;
    }
}

std::string SMP_Test::createBigFloat(uint64_t preRadix, uint64_t postRadix, char digit)
{
    std::string bigFloat{};

    if (preRadix > 0)
    {
        for (uint64_t i = 0; i <= preRadix; i++)
        {
            bigFloat.push_back(digit);
        }
        bigFloat.push_back('.');
    }
    else
    {
        bigFloat.push_back('0');
        bigFloat.push_back('.');
    }

    for (uint64_t i = 0; i <= postRadix; i++)
    {
        bigFloat.push_back(digit);
    }

    return bigFloat;
}