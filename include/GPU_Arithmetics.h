#ifndef GPU_ARITHMETICS_H
#define GPU_ARITHMETICS_H

#include "IArithmetics.h"

#include <vector>
#include <cstdint>

class GPU_Arithmetics : public IArithmetics
{
public:
    GPU_Arithmetics();
    SMP_Float add(const SMP_Float &op1, const SMP_Float &op2) override;
    SMP_Float multiply(const SMP_Float &op1, const SMP_Float &op2) override;

private:
    /**
     * Checks Carry Array if there are still Carry left to be added
     * Parameter:
     *       carry: Carry Array   
     *       size: size of expected Result 
     */
    bool checkForCarry(uint8_t *carry, uint64_t size);
    bool checkForCarry(uint64_t *carry, uint64_t size);
};

#endif