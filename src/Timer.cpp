#include "../include/Timer.h"

void Timer::start()
{
    startTime = std::chrono::steady_clock::now();
}

int64_t Timer::stop()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startTime).count();
}