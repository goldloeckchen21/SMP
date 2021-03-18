#ifndef CPU_ARITHMETICS_H
#define CPU_ARITHMETICS_H

#include "IArithmetics.h"

#include <vector>
#include <cstdint>

class CPU_Arithmetics : public IArithmetics
{
public:
    SMP_Float add(const SMP_Float &operand1, const SMP_Float &operand2) override;
    SMP_Float multiply(const SMP_Float &operand1, const SMP_Float &operand2) override;

private:
    bool checkForCarry(std::vector<uint64_t> &carry, uint64_t size);
};

#endif