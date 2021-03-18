#ifndef SMP_FLOAT_H
#define SMP_FLOAT_H

#include <string>
#include <vector>
#include <cstdint>

class SMP_Float
{
public:
    SMP_Float(std::string input);
    SMP_Float(const std::vector<uint8_t> &resultDigits, int64_t exp1, int64_t exp2, bool multiplication);

    void normalizeMantissa(std::string toNorm);  // Normalizes input string to: <mantissa>x10e<exponent>. Leading "0." is implicit.
    void convertMantissaToVector();              // Converts normalized mantissa string to an uint8_t vector
    uint64_t getLeadingZeros(std::string input); // Returns amount of leading zeros after radix point
    void printStatus();                          // Prints object status information
    void printDigits();

    int64_t exponent{0};
    uint64_t lenMantissa{0};
    std::string mantissa{};
    std::vector<uint8_t> digits{};
};

#endif