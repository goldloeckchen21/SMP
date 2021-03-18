#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <cstdint>

class Timer
{
public:
    void start();
    int64_t stop();
private:
    std::chrono::steady_clock::time_point startTime{};
    std::chrono::steady_clock::time_point endTime{};
};

#endif