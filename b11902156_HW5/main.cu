#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

const int N = 1024;
const int MAX = 1000000;
const double eps = 1.0e-1;
float *h_a, *h_u[2], *d_u[2];

__global__ void heatDiffusion(const float* u_old, float* u_new, float* C, int thread_id, int thread_count) {
    extern __shared__ float cache[];

    float t, l, r, b, diff = 0.0;
    int Nx = blockDim.x * gridDim.x;
    int Ny = blockDim.y * gridDim.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int cacheIndex = threadIdx.x + blockDim.x * threadIdx.y;
    int site = x + Nx * y + thread_id * Nx * Ny;

    if (x && x < Nx - 1 && (thread_id || y) && (thread_id < thread_count - 1 || y < Ny - 1)) {
        l = u_old[site - 1];
        r = u_old[site + 1];
        b = u_old[site - Nx];
        t = u_old[site + Nx];
        u_new[site] = 0.25 * (b + t + l + r);
        diff = u_new[site] - u_old[site];
    }

    cache[cacheIndex] = diff * diff;
    __syncthreads();

    int ib = blockDim.x * blockDim.y / 2;
    while (ib != 0) {
        if (cacheIndex < ib) {
            cache[cacheIndex] += cache[cacheIndex + ib];
        }
        __syncthreads();
        ib /= 2;
    }
    
    if (cacheIndex == 0) {
        C[blockIdx.x + gridDim.x * blockIdx.y] = cache[0];
    }
}

int main(void) {
    int NGPU;
    printf("Enter the number of GPU:");
    scanf("%d", &NGPU);
    if (NGPU <= 0 || NGPU > 2) {
        printf("Number of GPU must be less or equal to 2\n");
        exit(0);
    }
    printf("%d\n", NGPU);

    h_a = (float*)malloc(N * N * sizeof(float));

    int *dev = (int*)malloc(NGPU * sizeof(int));
    for (int i = 0; i < NGPU; i++) {
        dev[i] = i;
    }

    int cpu_thread_id = 0;

    int tx, ty;
    printf("Enter the number of threads (tx, ty):");
    scanf("%d%d", &tx, &ty);
    printf("%d %d\n", tx, ty);

    int threadsPerBlock = tx * ty;

    printf("Tumber of threads per block is %d\n", threadsPerBlock);
    if (threadsPerBlock > 1024) {
        printf("The number of threads per block must be less than 1024 ! \n");
        exit(0);
    }

    dim3 threads(tx, ty);

    int bx = N / tx;
    int by = N / ty / NGPU;
    if (bx * tx != N || by * ty * NGPU != N) {
        printf("The block size if incorrect\n");
        exit(0);
    }

    int blocksPerGrid = bx * by;
    printf("The number of blocks is %d\n", blocksPerGrid);

    if (blocksPerGrid > 2147483647) {
        printf("The number of blocks must be less than 2147483647 ! \n");
        exit(0);
    }

    dim3 blocks(bx, by);
    printf("The dimension of the grid is (%d, %d)\n", bx, by);

    int size = N * N * sizeof(float);
    int sb = blocksPerGrid * sizeof(float);

    h_u[0] = (float*)malloc(size);
    h_u[1] = (float*)malloc(size);

    if (!h_u[0] || !h_u[1]) {
        printf("Not enough of memory\n");
        exit(1);
    }
    memset(h_u[0], 0, sizeof(h_u[0]));
    memset(h_u[1], 0, sizeof(h_u[1]));

    for (int i = 0; i < N; i++) {
        h_u[0][i] = 273.0;
        h_u[1][i] = 273.0;
        h_u[0][i * N] = 273.0;
        h_u[1][i * N] = 273.0;
        h_u[0][(N - 1) + i * N] = 273.0;
        h_u[1][(N - 1) + i * N] = 273.0;
        h_u[0][i + (N - 1) * N] = 400.0;
        h_u[1][i + (N - 1) * N] = 400.0;
    }

    /*
    FILE *out1 = fopen("initial.dat", "w");
    fprintf(out1, "intial configuration:\n");
    for (int j = N - 1; j >= 0; j--) {
        for (int i = 0; i < N; i++) {
            fprintf(out1, "%.2e ", h_u[0][i + j * N]);
        }
        fprintf(out1, "\n");
    }
    fclose(out1);
    */

    cudaEvent_t start, stop;

    float InTime, gpuTime, OutTime;
    float error = eps + 1.0;
    float *errs = (float*)malloc(NGPU * sizeof(float));
    int iter = 0;

    omp_set_num_threads(NGPU);

#pragma omp parallel private(cpu_thread_id)
{   
    cpu_thread_id = omp_get_thread_num();
    cudaSetDevice(dev[cpu_thread_id]);

    if (cpu_thread_id == 0) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);   
        cudaEventRecord(start, 0);
    }

    float *h_C, *d_C;
    h_C = (float*)malloc(sb);
    cudaMalloc((void**)&d_C, sb);

    if (cpu_thread_id == 0) {
        cudaMallocManaged((void**)&d_u[0], size);
        cudaMallocManaged((void**)&d_u[1], size);

        cudaMemcpy(d_u[0], h_u[0], size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_u[1], h_u[1], size, cudaMemcpyHostToDevice);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&InTime, start, stop);
        printf("Input time for GPU: %f (ms) \n", InTime);

        cudaEventRecord(start, 0);        
    }
#pragma omp barrier

    int sm = threadsPerBlock * sizeof(float);
    while (iter < MAX && error > eps) {
        int turn = iter & 1;
        heatDiffusion<<<blocks, threads, sm>>>(d_u[turn], d_u[!turn], d_C, cpu_thread_id, NGPU);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);

        errs[cpu_thread_id] = 0.0;
        for (int i = 0; i < blocksPerGrid; i++) {
            errs[cpu_thread_id] += h_C[i];
        }

#pragma omp barrier
        if (cpu_thread_id == 0) {
            error = 0.0;
            for (int i = 0; i < NGPU; i++) {
                error += errs[i];
            }
            error = sqrt(error);
            iter++;
        }
#pragma omp barrier
    }
    
    if (cpu_thread_id == 0) {
        printf("error (GPU) = %.15e\n",error);
        printf("total iterations (GPU) = %d\n",iter);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        printf("Processing time for GPU: %f (ms) \n", gpuTime);
        printf("GPU Gflops: %f\n", 7.0 * (N - 2) * (N - 2) * iter / (1000000.0 * gpuTime));
        
        cudaEventRecord(start, 0);
    }

    cudaMemcpy(h_a + N * N / NGPU * cpu_thread_id, d_u[iter & 1], size / NGPU, cudaMemcpyDeviceToHost);

    if (cpu_thread_id == 0) {
        cudaFree(d_u[0]);
        cudaFree(d_u[1]);        
    }

    cudaFree(d_C);
    free(h_C);

    if (cpu_thread_id == 0) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&OutTime, start, stop);
        printf("Output time for GPU: %f (ms) \n", OutTime);
    }
}

    float gputime_tot;
    gputime_tot = InTime + gpuTime + OutTime;
    printf("Total time for GPU: %f (ms) \n", gputime_tot);
    
    cudaEventRecord(start,0);

      // to compute the reference solution

    error = eps + 1;      // any value bigger than eps 
    iter = 0;            // counter for iterations
    double diff; 

    float t, l, r, b;    // top, left, right, bottom

    while ((error > eps) && (iter < MAX) ) {
        int i = iter & 1;
        error = 0.0;
        for(int y=0; y<N; y++) {
            for(int x=0; x<N; x++) {
                if (x && x < N - 1 && y && y < N - 1) {
                    int site = x+y*N;
                    b = h_u[i][site - 1]; 
                    l = h_u[i][site + 1]; 
                    r = h_u[i][site - N]; 
                    t = h_u[i][site + N]; 
                    h_u[!i][site] = 0.25*(b+l+r+t);
                    diff = h_u[!i][site]-h_u[i][site]; 
                    error = error + diff*diff;
                }
            } 
        }
        iter++;
        error = sqrt(error);
    }                   // exit if error < eps
    
    printf("error (CPU) = %.15e\n",error);
    printf("total iterations (CPU) = %d\n",iter);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime(&cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    double flops = 7.0*(N-2)*(N-2)*iter;
    printf("CPU Gflops: %lf\n",flops/(1000000.0*cputime));

    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));
    fflush(stdout);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(dev);
    free(h_u[0]);
    free(h_u[1]);
    free(h_a);
        
    /*
    FILE *out2 = fopen("gpu.dat", "w");
    fprintf(out2, "intial configuration:\n");
    for (int j = N - 1; j >= 0; j--) {
        for (int i = 0; i < N; i++) {
            fprintf(out1, "%.2e ", h_u[0][i + j * N]);
        }
        fprintf(out2, "\n");
    }
    fclose(out2);
    */

    cudaDeviceReset();
}