#include "include/SMP_Test.h"
#include "include/SMP_Float.h"
#include "include/IArithmetics.h"
#include "include/CPU_Arithmetics.h"
#include "include/GPU_Arithmetics.h"

#include <chrono>
#include <string>
#include <iostream>

int main()
{
    IArithmetics *cpu = new CPU_Arithmetics();
    IArithmetics *gpu = new GPU_Arithmetics();

    SMP_Test tester;

    auto timeElapsed = tester.executeTests(cpu, false);
    std::cout << "\u001b[35mDuration CPU: \u001b[0m" << timeElapsed << " ms" << std::endl;
    std::cout << "--------------- FINISHED CPU TEST ---------------" << std::endl << std::endl;

    timeElapsed = tester.executeTests(gpu, false);
    std::cout << "\u001b[35mDuration GPU: \u001b[0m" << timeElapsed << " ms" << std::endl;
    std::cout << "--------------- FINISHED GPU TEST ---------------" << std::endl << std::endl;

    delete cpu;
    delete gpu;

    return EXIT_SUCCESS;
}