#include <stdio.h>
#include <stdlib.h>
// nvcc -arch=compute_61 "-code=sm_61,sm_61" -O2 -m64 main.cu

const int N = 6400;
float *h_A, *h_B, *h_C, *h_D;
float *d_A, *d_B, *d_C;

void randomInit(float*, int);

__global__ void VecAdd(const float* A, const float* B, float* C, const int N) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N && j < N) {
        int idx = i * N + j;
        C[idx] = 1.0 / A[idx] + 1.0 / B[idx];
    }
    __syncthreads();
}

int main() {
    int gid = 0;
    if (cudaSetDevice(gid) != cudaSuccess) {
        printf("GPU failed\n");
        exit(1);
    }

    h_A = (float*)malloc(N * N * sizeof(float));
    h_B = (float*)malloc(N * N * sizeof(float));
    h_C = (float*)malloc(N * N * sizeof(float));

    randomInit(h_A, N * N);
    randomInit(h_B, N * N);

    dim3 blockDimension;
    blockDimension.z = 1;
loop:
    printf("Enter the dimension for each block: ");
    scanf("%d%d", &blockDimension.x, &blockDimension.y);
    printf("%d %d\n", blockDimension.x, blockDimension.y);
    int threadsPerBlock = blockDimension.x * blockDimension.y;
    if (threadsPerBlock > 1024) {
        printf("The number of threads per block must be less than 1024 ! \n");
        goto loop;
    }

    dim3 gridDimension;
    gridDimension.x = (N + blockDimension.x - 1) / blockDimension.x;
    gridDimension.y = (N + blockDimension.y - 1) / blockDimension.y;
    gridDimension.z = 1;

    int blocksPerGrid = gridDimension.x * gridDimension.y;
    printf("The dimension of the grid is %dx%d\n", gridDimension.x, gridDimension.y);
    printf("The number of blocks is %d\n", blocksPerGrid);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float InTime;
    cudaEventElapsedTime(&InTime, start, stop);
    printf("Input time for GPU: %f (ms) \n", InTime);

    cudaEventRecord(start, 0);

    VecAdd<<<gridDimension, blockDimension>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("Processing time for GPU: %f (ms) \n", gpuTime);
    printf("GPU Gflops: %f\n", 3 * N * N / (1000000.0 * gpuTime));

    cudaEventRecord(start, 0);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float OutTime;
    cudaEventElapsedTime(&OutTime, start, stop);
    printf("Output time for GPU: %f (ms) \n", OutTime);

    float gputime_tot;
    gputime_tot = InTime + gpuTime + OutTime;
    printf("Total time for GPU: %f (ms) \n", gputime_tot);

    cudaEventRecord(start, 0);

    h_D = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N * N; i++) {
        h_D[i] = 1.0 / h_A[i] + 1.0 / h_B[i];
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cpuTime;
    cudaEventElapsedTime(&cpuTime, start, stop);
    printf("Processing time for CPU: %f (ms) \n", cpuTime);
    printf("CPU Gflops: %f\n", 3 * N * N / (1000000.0 * cpuTime));
    printf("Speed up of GPU = %f\n\n", cpuTime / (gputime_tot));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Check result:\n");
    bool flag = true;
    //double sum = 0, diff;
    for (int i = 0; i < N * N; i++) {
        if (h_D[i] != h_C[i]) flag = false;
        //diff = abs(h_D[i] - h_C[i]);
        //sum += diff * diff;
    }
    //sum = sqrt(sum);
    //printf("norm(h_C - h_D)=%20.15e\n\n", sum);
    printf("%s\n", (flag)? "Correct": "Incorrect");

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    cudaDeviceReset();
}

void randomInit(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = rand() / (float)RAND_MAX;
    }
}