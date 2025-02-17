
#include <stdio.h>

__global__ void add_arrays(float* x, float* y, float* out, int N, int clockRate) {
    int index = threadIdx.x;

    if (index < N) {
        int start_time = clock();
        int delay_in_ms = 10;
        int delay_in_clock_cycles = delay_in_ms * (clockRate / 1000);

        while (clock() - start_time < delay_in_clock_cycles) {
            // Wait for 10ms
        }

        out[index] = x[index] + y[index];
    }
}