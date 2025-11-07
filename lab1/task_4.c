#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define __USE_XOPEN
#include <math.h>
#include <string.h>

double F(double x, double y) {
    return 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
}

// Аналитическое решение для заданного f и граничного значения border_val
double UExactFunction(double x, double y, double border_val) {
    return border_val - sin(M_PI * x) * sin(M_PI * y);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int my_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    int max_iter = 100000;
    double tolerance = 1e-6;

    if (argc < 2) {
        if (my_rank == 0) {
            printf("Параметры: %s <grid_size_N> [border_val] [max_iter]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int grid_size = atoi(argv[1]);
    double border_val = 0.0;
    if (argc >= 3) {
        border_val = atof(argv[2]);
    }
    if (argc >= 4) {
        max_iter = atoi(argv[3]);
    }

    if (grid_size % comm_size != 0) {
        if (my_rank == 0) {
            printf("Размер сетки должен делиться на количество процессов\n");
        }
        MPI_Finalize();
        return 1;
    }

    int local_grid_size = grid_size / comm_size;
    int start_row_global = my_rank * local_grid_size;
    double grid_step = 1.0 / (grid_size + 1);

    // Размер локального блока (rows x cols) включая фиктивные строки и граничные столбцы
    int rows = local_grid_size + 2;
    int cols = grid_size + 2;
    size_t block_size = (size_t)rows * cols;

    double *u = (double *)malloc(block_size * sizeof(double));
    double *u_new = (double *)malloc(block_size * sizeof(double));
    if (!u || !u_new) {
        fprintf(stderr, "my_rank %d: аллокация провалилась\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Инициализация (включая внешние фиктивные строки и столбцы) значением border_val
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            u[(i)*cols + (j)] = border_val;
            u_new[(i)*cols + (j)] = border_val;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    char is_converged = 0;
    for (int iter = 0; iter < max_iter && !is_converged; ++iter) {
        // Выполняем один полный волновой проход антидиагонали anti_diag = i + j,
        for (int anti_diag = 2; anti_diag <= 2 * grid_size; ++anti_diag) {
            for (int i_local = 1; i_local <= local_grid_size; ++i_local) {
                int i_global = start_row_global + i_local;
                int j_global = anti_diag - i_global;
                if (j_global >= 1 && j_global <= grid_size) {
                    int j = j_global;
                    // Используем Гаусса-Зейделя в волновой схеме:
                    // новые значения сверху и слева (u_new), старые справа и снизу (u)
                    double up = u_new[(i_local - 1) * cols + (j)];
                    double down = u[(i_local + 1) * cols + (j)];
                    double left = u_new[i_local * cols + j - 1];
                    double right = u[i_local * cols + j + 1];
                    double x = j * grid_step;
                    double y = i_global * grid_step;
                    double fval = F(x, y);

                    double newval =
                        0.25 * (up + down + left + right - grid_step * grid_step * fval);
                    u_new[i_local * cols + j] = newval;
                }
            }

            /* После вычисления текущей антидиагонали anti_diag нужно обменять
               обновлённые внутренние строки (i_local=1 и i_local=local_n)
               с соседними процессами, чтобы они могли использовать свежие u_new.
               Используем симметричные MPI_Sendrecv. */
            int tag = anti_diag;  // тэг зависит от anti_diag - гарантирует согласованность по шагу

            // Обмен с верхним соседом: каждый процесс, у которого my_rank>0, отправляет
            // свою внутреннюю строку i_local=1 вверх и принимает в фиктивную-строку u_new[0].
            if (my_rank > 0) {
                MPI_Sendrecv(&u_new[cols], cols, MPI_DOUBLE, my_rank - 1, tag, &u_new[0], cols,
                             MPI_DOUBLE, my_rank - 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // верхняя граница пластины - фиксированное значение border_val
                for (int j = 0; j < cols; ++j) {
                    u_new[j] = border_val;
                }
            }

            // Обмен с нижним соседом: отправляем свою внутреннюю строку i_local=local_grid_size
            // вниз и принимаем в фиктивную строку u_new[local_grid_size+1].
            if (my_rank < comm_size - 1) {
                MPI_Sendrecv(&u_new[local_grid_size * cols], cols, MPI_DOUBLE, my_rank + 1, tag,
                             &u_new[(local_grid_size + 1) * cols], cols, MPI_DOUBLE, my_rank + 1,
                             tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // нижняя граница пластины - фиксированное значение border_val
                for (int j = 0; j < cols; ++j) {
                    u_new[(local_grid_size + 1) * cols + j] = border_val;
                }
            }
        }

        // Проверка сходимости: сравниваем u_new и u на внутренних точках
        double local_max_diff = 0.0;
        for (int i_local = 1; i_local <= local_grid_size; ++i_local) {
            for (int j = 1; j <= grid_size; ++j) {
                double diff = fabs(u_new[i_local * cols + j] - u[i_local * cols + j]);
                if (diff > local_max_diff) {
                    local_max_diff = diff;
                }
            }
        }

        double global_max_diff;
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // swap pointers для следующей итерации
        double *tmp = u;
        u = u_new;
        u_new = tmp;

        if (my_rank == 0 && (iter % 100 == 0 || iter == 0)) {
            printf("Итерация %d: max diff = %.6e\n", iter, global_max_diff);
        }

        if (global_max_diff < tolerance) {
            is_converged = 1;
            if (my_rank == 0) {
                printf("Сошлась после %d итераций\n", iter + 1);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    // Теперь рассчитываем ошибку относительно аналитического решения
    // В массиве u находятся актуальные значения после последнего swap
    double local_max_err = 0.0;
    for (int i_local = 1; i_local <= local_grid_size; ++i_local) {
        int i_global = start_row_global + i_local;
        double y = i_global * grid_step;
        for (int j = 1; j <= grid_size; ++j) {
            double x = j * grid_step;
            double ua = u[i_local * cols + j];
            double ue = UExactFunction(x, y, border_val);
            double err = fabs(ua - ue);
            if (err > local_max_err) {
                local_max_err = err;
            }
        }
    }
    double global_max_err;
    MPI_Allreduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("grid_size=%d, processes=%d, time=%.6f секунд, global_max_err= %.6e\n", grid_size,
               comm_size, t_end - t_start, global_max_err);
    }

    free(u);
    free(u_new);
    MPI_Finalize();
    return 0;
}
