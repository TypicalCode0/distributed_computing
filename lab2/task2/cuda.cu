#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "time.h"

const double G = 6.67430e-11;
const double delta_t = 0.01;

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (Code: %d)\n",
                cudaGetErrorString(result), result);
        exit(1);
    }
}

__global__ void reset_forces(double* fx, double* fy, double* fz, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        fx[i] = 0.0;
        fy[i] = 0.0;
        fz[i] = 0.0;
    }
}

__global__ void compute_forces(const double* m, const double* x, const double* y, const double* z,
                               double* fx, double* fy, double* fz, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    double pos_xi = x[i];
    double pos_yi = y[i];
    double pos_zi = z[i];
    double mass_i = m[i];

    double f_ix = 0.0;
    double f_iy = 0.0;
    double f_iz = 0.0;

    for (int j = i + 1; j < n; ++j) {
        double dx = x[j] - pos_xi;
        double dy = y[j] - pos_yi;
        double dz = z[j] - pos_zi;

        double distSq = dx*dx + dy*dy + dz*dz;
        double dist = sqrt(distSq);

        if (dist < 1e-9) continue;

        double distCubed = dist * dist * dist;
        double f_mag = G * mass_i * m[j] / distCubed;

        double force_x = f_mag * dx;
        double force_y = f_mag * dy;
        double force_z = f_mag * dz;

        f_ix += force_x;
        f_iy += force_y;
        f_iz += force_z;

        atomicAdd(&fx[j], -force_x);
        atomicAdd(&fy[j], -force_y);
        atomicAdd(&fz[j], -force_z);
    }

    atomicAdd(&fx[i], f_ix);
    atomicAdd(&fy[i], f_iy);
    atomicAdd(&fz[i], f_iz);
}

__global__ void update_particles(double* m, double* x, double* y, double* z,
                                 double* vx, double* vy, double* vz,
                                 const double* fx, const double* fy, const double* fz,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double mass = m[i];

        double ax = fx[i] / mass;
        double ay = fy[i] / mass;
        double az = fz[i] / mass;

        x[i] += vx[i] * delta_t;
        y[i] += vy[i] * delta_t;
        z[i] += vz[i] * delta_t;

        vx[i] += ax * delta_t;
        vy[i] += ay * delta_t;
        vz[i] += az * delta_t;
    }
}

int main(int argc, char* argv[]) {
    double start_time, end_time;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <tend> <filename>\n", argv[0]);
        return 1;
    }

    double t_end = atof(argv[1]);
    char* filename = argv[2];
    int n;

    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return 1;
    }
    if (fscanf(file, "%d", &n) != 1) {
        fprintf(stderr, "Error reading number of particles\n");
        return 1;
    }

    double *h_m = (double*)malloc(n * sizeof(double));
    double *h_x = (double*)malloc(n * sizeof(double));
    double *h_y = (double*)malloc(n * sizeof(double));
    double *h_z = (double*)malloc(n * sizeof(double));
    double *h_vx = (double*)malloc(n * sizeof(double));
    double *h_vy = (double*)malloc(n * sizeof(double));
    double *h_vz = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; ++i) {
        fscanf(file, "%lf %lf %lf %lf %lf %lf %lf",
               &h_m[i],
               &h_x[i],
               &h_y[i],
               &h_z[i],
               &h_vx[i],
               &h_vy[i],
               &h_vz[i]);
    }
    fclose(file);

    FILE* outFile = fopen("output.csv", "w");
    if (!outFile) {
        fprintf(stderr, "Error creating output file\n");
        return 1;
    }

    double *d_m, *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_fx, *d_fy, *d_fz;
    checkCuda(cudaMalloc(&d_m, n * sizeof(double)));
    checkCuda(cudaMalloc(&d_x, n * sizeof(double)));
    checkCuda(cudaMalloc(&d_y, n * sizeof(double)));
    checkCuda(cudaMalloc(&d_z, n * sizeof(double)));
    checkCuda(cudaMalloc(&d_vx, n * sizeof(double)));
    checkCuda(cudaMalloc(&d_vy, n * sizeof(double)));
    checkCuda(cudaMalloc(&d_vz, n * sizeof(double)));
    checkCuda(cudaMalloc(&d_fx, n * sizeof(double)));
    checkCuda(cudaMalloc(&d_fy, n * sizeof(double)));
    checkCuda(cudaMalloc(&d_fz, n * sizeof(double)));

    checkCuda(cudaMemcpy(d_m, h_m, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_x, h_x, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_y, h_y, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_z, h_z, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_vx, h_vx, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_vy, h_vy, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_vz, h_vz, n * sizeof(double), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    double t = 0.0;

    GET_TIME(start_time);
    while (t <= t_end) {
        checkCuda(cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(h_y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        fprintf(outFile, "%.6f", t);
        for (int i = 0; i < n; ++i) {
            fprintf(outFile, " %.6f %.6f", h_x[i], h_y[i]);
        }
        fprintf(outFile, "\n");

        if (t >= t_end) break;

        reset_forces<<<blocksPerGrid, threadsPerBlock>>>(d_fx, d_fy, d_fz, n);
        checkCuda(cudaGetLastError());

        compute_forces<<<blocksPerGrid, threadsPerBlock>>>(d_m, d_x, d_y, d_z, d_fx, d_fy, d_fz, n);
        checkCuda(cudaGetLastError());

        update_particles<<<blocksPerGrid, threadsPerBlock>>>(d_m, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, n);
        checkCuda(cudaGetLastError());

        t += delta_t;
    }

    GET_TIME(end_time);

    printf("%lf\n", end_time - start_time);

    fclose(outFile);

    free(h_m);
    free(h_x);
    free(h_y);
    free(h_z);
    free(h_vx);
    free(h_vy);
    free(h_vz);

    cudaFree(d_m);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);

    return 0;
}