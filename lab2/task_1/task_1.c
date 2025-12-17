/*
компиляция
gcc task_1.c -o task_1 -fopenmp -lm
запуск
./task_1 1 1000 (пример)
для тестирования и графиков:
./run_tests_1.sh
python3 plot_graphs_1.py
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <omp.h>
#include <limits.h>
#include <errno.h>

typedef struct { double x, y; } Point;

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Использование: %s <nthreads> <npoints_side> [max_iter]\n", argv[0]);
        return EXIT_FAILURE;
    }

    errno = 0;
    long user_nthreads = strtol(argv[1], NULL, 10);
    if (errno || user_nthreads <= 0) {
        fprintf(stderr, "Ошибка: некорректное количество потоков: %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    errno = 0;
    unsigned long npoints_side_ul = strtoul(argv[2], NULL, 10);
    if (errno || npoints_side_ul == 0) {
        fprintf(stderr, "Ошибка: некорректное количество точек на сторону: %s\n", argv[2]);
        return EXIT_FAILURE;
    }
    size_t npoints_side = (size_t)npoints_side_ul;

    int max_iter = 1000;
    if (argc == 4) {
        errno = 0;
        long t = strtol(argv[3], NULL, 10);
        if (errno || t <= 0) {
            fprintf(stderr, "Ошибка: некорректный max_iter: %s\n", argv[3]);
            return EXIT_FAILURE;
        }
        max_iter = (int)t;
    }

    if (npoints_side > 0 && npoints_side > SIZE_MAX / npoints_side) {
        fprintf(stderr, "Ошибка: слишком большой npoints_side, возможна арифметическая переполнения.\n");
        return EXIT_FAILURE;
    }
    size_t capacity = npoints_side * npoints_side;

    omp_set_num_threads((int)user_nthreads);

    const double x_min = -2.0, x_max = 1.0;
    const double y_min = -1.5, y_max = 1.5;

    double dx = (npoints_side > 1) ? (x_max - x_min) / (double)(npoints_side - 1) : 0.0;
    double dy = (npoints_side > 1) ? (y_max - y_min) / (double)(npoints_side - 1) : 0.0;

    Point *mandelbrot_points = NULL;
    if (capacity > 0) {
        mandelbrot_points = malloc(capacity * sizeof(Point));
        if (!mandelbrot_points) {
            fprintf(stderr, "Ошибка: не удалось выделить память для %zu точек (%.2f MB).\n",
                    capacity, (double)capacity * sizeof(Point) / (1024.0*1024.0));
            return EXIT_FAILURE;
        }
    }

    size_t point_count = 0;

    double t_start = omp_get_wtime();

    #pragma omp parallel
    {
        int nth = omp_get_num_threads();
        int tid = omp_get_thread_num();

        size_t local_capacity = (nth > 0) ? (capacity / (size_t)nth + 16) : 1024;
        if (local_capacity == 0) local_capacity = 16; // защита от нуля
        Point *local_buf = malloc(local_capacity * sizeof(Point));
        if (!local_buf) {
            #pragma omp critical
            {
                fprintf(stderr, "Поток %d: не удалось выделить локальный буфер (size=%zu)\n", tid, local_capacity);
            }
            local_capacity = 0;
        }
        size_t local_cnt = 0;

        const int chunk = 64;
        #pragma omp for collapse(2) schedule(dynamic, chunk)
        for (size_t i = 0; i < npoints_side; ++i) {
            for (size_t j = 0; j < npoints_side; ++j) {
                double x = x_min + (double)i * dx;
                double y = y_min + (double)j * dy;

                double zx = 0.0, zy = 0.0;
                int iter;
                for (iter = 0; iter < max_iter; ++iter) {
                    double zx2 = zx * zx;
                    double zy2 = zy * zy;
                    double zx_new = zx2 - zy2 + x;
                    double zy_new = 2.0 * zx * zy + y;

                    if (zx_new * zx_new + zy_new * zy_new > 4.0) break;

                    zx = zx_new;
                    zy = zy_new;
                }

                if ((size_t)iter >= (size_t)max_iter) {
                    if (local_capacity == 0) continue;
                    if (local_cnt >= local_capacity) {
                        size_t newcap = local_capacity * 2;
                        Point *tmp = realloc(local_buf, newcap * sizeof(Point));
                        if (!tmp) {
                            #pragma omp critical
                            {
                                fprintf(stderr, "Поток %d: realloc не удался при расширении до %zu\n", tid, newcap);
                            }
                            continue;
                        }
                        local_buf = tmp;
                        local_capacity = newcap;
                    }
                    local_buf[local_cnt].x = x;
                    local_buf[local_cnt].y = y;
                    local_cnt++;
                }
            }
        }

        if (local_cnt > 0) {
            size_t start_idx;
            #pragma omp atomic capture
            { start_idx = point_count; point_count += local_cnt; }

            if (start_idx + local_cnt <= capacity) {
                for (size_t k = 0; k < local_cnt; ++k) {
                    mandelbrot_points[start_idx + k] = local_buf[k];
                }
            } else {
                #pragma omp critical
                {
                    fprintf(stderr, "Поток %d: переполнение глобального буфера: start=%zu local=%zu capacity=%zu\n",
                            tid, start_idx, local_cnt, capacity);
                }
            }
        }

        free(local_buf);
    }

    double t_end = omp_get_wtime();

    printf("Время вычислений (только параллельный цикл): %.6f сек\n", t_end - t_start);
    printf("Найдено %zu точек.\n", point_count);

    FILE *out = fopen("mandelbrot_points.csv", "w");
    if (!out) {
        fprintf(stderr, "Ошибка: не удалось открыть файл mandelbrot_points.csv для записи.\n");
        free(mandelbrot_points);
        return EXIT_FAILURE;
    }
    fprintf(out, "real,imaginary\n");
    for (size_t i = 0; i < point_count; ++i) {
        fprintf(out, "%.12g,%.12g\n", mandelbrot_points[i].x, mandelbrot_points[i].y);
    }
    fclose(out);

    free(mandelbrot_points);

    return EXIT_SUCCESS;
}