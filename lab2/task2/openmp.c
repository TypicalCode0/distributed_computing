#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "timer.h"

const double G = 6.67430e-11;
const double delta_t = 0.01;

typedef struct {
    double m;
    double x, y, z;
    double vx, vy, vz;
    double fx, fy, fz;
} Particle;

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


    Particle* particles = (Particle*)malloc(n * sizeof(Particle));
    for (int i = 0; i < n; ++i) {
        // m, x, y, z, vx, vy, vz
        fscanf(file, "%lf %lf %lf %lf %lf %lf %lf",
               &particles[i].m,
               &particles[i].x,
               &particles[i].y,
               &particles[i].z,
               &particles[i].vx,
               &particles[i].vy,
               &particles[i].vz);

        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
        particles[i].fz = 0.0;
    }
    fclose(file);

    FILE* outFile = fopen("output.csv", "w");
    if (!outFile) {
        fprintf(stderr, "Error creating output file\n");
        return 1;
    }

    double t = 0.0;

    GET_TIME(start_time);
    while (t <= t_end) {
        fprintf(outFile, "%.6f", t);
        for (int i = 0; i < n; ++i) {
            fprintf(outFile, " %.6f %.6f", particles[i].x, particles[i].y);
        }
        fprintf(outFile, "\n");

        if (t >= t_end) break;
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            particles[i].fx = 0.0;
            particles[i].fy = 0.0;
            particles[i].fz = 0.0;
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double f_ix = 0.0, f_iy = 0.0, f_iz = 0.0;

            for (int j = i + 1; j < n; ++j) {
                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;
                double dz = particles[j].z - particles[i].z;

                double distSq = dx*dx + dy*dy + dz*dz;
                double dist = sqrt(distSq);

                if (dist < 1e-9) continue;

                double distCubed = dist * dist * dist;
                double f_mag = G * particles[i].m * particles[j].m / distCubed;

                double fx = f_mag * dx;
                double fy = f_mag * dy;
                double fz = f_mag * dz;

                f_ix += fx;
                f_iy += fy;
                f_iz += fz;

                #pragma omp atomic
                particles[j].fx -= fx;
                #pragma omp atomic
                particles[j].fy -= fy;
                #pragma omp atomic
                particles[j].fz -= fz;
            }

            #pragma omp atomic
            particles[i].fx += f_ix;
            #pragma omp atomic
            particles[i].fy += f_iy;
            #pragma omp atomic
            particles[i].fz += f_iz;
        }

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            double ax = particles[i].fx / particles[i].m;
            double ay = particles[i].fy / particles[i].m;
            double az = particles[i].fz / particles[i].m;

            particles[i].x += particles[i].vx * delta_t;
            particles[i].y += particles[i].vy * delta_t;
            particles[i].z += particles[i].vz * delta_t;

            particles[i].vx += ax * delta_t;
            particles[i].vy += ay * delta_t;
            particles[i].vz += az * delta_t;
        }

        t += delta_t;
    }
    GET_TIME(end_time);

    printf("%lf\n", end_time - start_time);

    fclose(outFile);
    free(particles);
    return 0;
}