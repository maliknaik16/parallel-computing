
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <thread>

using namespace std;

#define ARRAY_SIZE 1000

__global__ void add_arrays(float*, float*, float*, int, int);

void add_arrays_cpu(float* x, float* y, float* z, int N) {
    for(int i = 0; i < N; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        z[i] = x[i] + y[i];
    }
}

int main() {

    std::chrono::duration<double> elapsed;
    std::chrono::high_resolution_clock::time_point start, end;
    float *a, *b, *c, *dummy;
    float *d_a, *d_b, *d_c;

    // Set fixed decimal format with precision
    std::cout << std::fixed << std::setprecision(6);

    cout << "Initializing some memory..." << endl;
    // // C-style memory allocation.
    // a = (float*)malloc(sizeof(float) * ARRAY_SIZE);
    // b = (float*)malloc(sizeof(float) * ARRAY_SIZE);
    // c = (float*)malloc(sizeof(float) * ARRAY_SIZE);
    // dummy = (float*)malloc(sizeof(float) * ARRAY_SIZE);

    // C++-style memory allocation.
    a = new float[ARRAY_SIZE];
    b = new float[ARRAY_SIZE];
    c = new float[ARRAY_SIZE];
    dummy = new float[ARRAY_SIZE];

    cout << "Assigning some values to arrays a and b" << endl;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = (float)(i + 1);
        b[i] = (float)2 * (i + 1);
    }

    cout << "Allocating memory on device memory (Nvidia GPU)" << endl;
    cudaMalloc((void**)&d_a, sizeof(float) * ARRAY_SIZE);
    cudaMalloc((void**)&d_b, sizeof(float) * ARRAY_SIZE);
    cudaMalloc((void**)&d_c, sizeof(float) * ARRAY_SIZE);

    cout << "Copying data from Host to Device memory" << endl;
    cudaMemcpy(d_a, a, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);

    cout << "Launch the kernel on 1 block with " << ARRAY_SIZE << " threads..." << endl;
    start = std::chrono::high_resolution_clock::now();
    int clockRate;
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);
    add_arrays<<<1, ARRAY_SIZE>>>(d_a, d_b, d_c, ARRAY_SIZE, clockRate);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    cout << "Time taken to add arrays on GPU is " << elapsed.count() << " seconds." << endl;

    cout << "Copy the results from Device to Host memory" << endl;
    cudaMemcpy(c, d_c, sizeof(float) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

    cout << "Printing first 5 results" << endl;
    for(int i = 0; i < 5; i++) {
        cout << "\t" << c[i] << endl;
    }

    cout << "Freeing up device memory..." << endl;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    start = std::chrono::high_resolution_clock::now();
    add_arrays_cpu(a, b, dummy, ARRAY_SIZE);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;    

    cout << "Time taken to add arrays on CPU is " << elapsed.count() << " seconds." << endl;
    // // C-style memory deallocation.
    // free(a);
    // free(b);
    // free(c);

    cout << "Freeing up host memory..." << endl;
    // C++ style memory deallocation.
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] dummy;
}
