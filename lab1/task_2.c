#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

// #define DEBUG

const int kRoot = 0;

double NextDouble() {
    return (double)rand() / RAND_MAX * 2.0 - 1.0;
}

void InitVector(double *vec, const int size) {
    for (int i = 0; i < size; ++i) {
        vec[i] = NextDouble();
    }
}

void PrintMat(double *mat, const int rows, const int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            printf("%.3f\t", mat[row * cols + col]);
        }
        printf("\n");
    }
    printf("##################################################################################\n");
}

void PrintVec(double *vec, const int size) {
    for (int row = 0; row < size; ++row) {
        printf("%.3f\t", vec[row]);
    }
    printf(
        "\n##################################################################################\n");
}

double MatVecInRowMode(double *mat, double *vec, double *res, const int rows, const int cols,
                       const int my_rank, const int comm_sz) {
    int rows_per_process[comm_sz];
    int base = rows / comm_sz;
    int rem = rows % comm_sz;
    for (int process = 0; process < comm_sz; ++process) {
        rows_per_process[process] = base + (process < rem ? 1 : 0);
    }

    int local_rows = rows_per_process[my_rank];
    double *local_mat = calloc(local_rows * cols, sizeof(double));

    int sendcount[comm_sz];
    for (int process = 0; process < comm_sz; ++process) {
        sendcount[process] = rows_per_process[process] * cols;
    }

    int displs[comm_sz];  // смещения (в количестве элементов)
    displs[0] = 0;
    for (int process = 1; process < comm_sz; ++process) {
        displs[process] = displs[process - 1] + sendcount[process - 1];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    MPI_Scatterv(mat, sendcount, displs, MPI_DOUBLE, local_mat, sendcount[my_rank], MPI_DOUBLE,
                 kRoot, MPI_COMM_WORLD);
    if (my_rank == kRoot) {
        MPI_Bcast(vec, cols, MPI_DOUBLE, kRoot, MPI_COMM_WORLD);
    } else {
        vec = calloc(cols, sizeof(double));
        MPI_Bcast(vec, cols, MPI_DOUBLE, kRoot, MPI_COMM_WORLD);
    }

    double *local_res = calloc(local_rows, sizeof(double));
    for (int row = 0; row < local_rows; ++row) {
        double sum = 0.0;
        double *mat_row = local_mat + (uint64_t)row * (uint64_t)cols;
        for (int col = 0; col < cols; ++col) {
            sum += mat_row[col] * vec[col];
        }
        local_res[row] = sum;
    }

    int displs_row[comm_sz];  // смещения (в количестве строк)
    displs_row[0] = 0;
    for (int process = 1; process < comm_sz; ++process) {
        displs_row[process] = displs_row[process - 1] + rows_per_process[process - 1];
    }
    MPI_Gatherv(local_res, local_rows, MPI_DOUBLE, res, rows_per_process, displs_row, MPI_DOUBLE,
                kRoot, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    free(local_mat);
    free(local_res);
    if (my_rank != kRoot) {
        free(vec);
    }

    double local_time = t_end - t_start;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return max_time;
}

double MatVecInColMode(double *mat, double *vec, double *res, const int rows, const int cols,
                       const int my_rank, const int comm_sz) {
    int base = cols / comm_sz;
    int rem = cols % comm_sz;
    int cols_per_process[comm_sz];
    for (int process = 0; process < comm_sz; ++process) {
        cols_per_process[process] = base + (process < rem ? 1 : 0);
    }
    int displs_col[comm_sz];
    displs_col[0] = 0;
    for (int process = 1; process < comm_sz; ++process) {
        displs_col[process] = displs_col[process - 1] + cols_per_process[process - 1];
    }
    int local_cols = cols_per_process[my_rank];

    double *sendbuf = NULL;
    int *sendcounts = NULL, *displs = NULL;
    if (my_rank == kRoot) {
        sendcounts = calloc(comm_sz, sizeof(int));
        displs = calloc(comm_sz, sizeof(int));
        for (int process = 0; process < comm_sz; ++process) {
            sendcounts[process] = cols_per_process[process] * rows;
        }
        displs[0] = 0;
        for (int process = 1; process < comm_sz; ++process) {
            displs[process] = displs[process - 1] + sendcounts[process - 1];
        }

        sendbuf = calloc(rows * cols, sizeof(double));  // храним столбцы как строки
        for (int process = 0; process < comm_sz; ++process) {
            int column_start = displs_col[process];
            int curr_local_cols = cols_per_process[process];
            double *dest = sendbuf + displs[process];
            for (int row = 0; row < rows; ++row) {
                const double *rowptr = mat + (uint64_t)row * cols + column_start;
                memcpy(dest + (uint64_t)row * curr_local_cols, rowptr,
                       sizeof(double) * curr_local_cols);
            }
        }
#ifdef DEBUG
        PrintMat(sendbuf, cols, rows);
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    double *local_mat = calloc(rows * local_cols, sizeof(double));
    MPI_Scatterv(sendbuf, sendcounts, displs, MPI_DOUBLE, local_mat, rows * local_cols, MPI_DOUBLE,
                 kRoot, MPI_COMM_WORLD);

    double *local_vec = calloc(local_cols, sizeof(double));
    MPI_Scatterv(vec, cols_per_process, displs_col, MPI_DOUBLE, local_vec, local_cols, MPI_DOUBLE,
                 kRoot, MPI_COMM_WORLD);

    double *local_res = calloc(rows, sizeof(double));
    for (int row = 0; row < rows; ++row) {
        double sum = 0.0;
        double *mat_row = local_mat + (uint64_t)row * local_cols;
        for (int col = 0; col < local_cols; ++col) {
            sum += mat_row[col] * local_vec[col];
        }
        local_res[row] = sum;
    }

    MPI_Reduce(local_res, res, rows, MPI_DOUBLE, MPI_SUM, kRoot, MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double total_time = t_end - t_start;

    if (my_rank == 0) {
        free(sendbuf);
        free(sendcounts);
        free(displs);
    }
    free(local_mat);
    free(local_vec);
    free(local_res);

    double max_time = 0.0;
    MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, kRoot, MPI_COMM_WORLD);
    return max_time;
}

double MatVecInBlockMode(double *mat, double *vec, double *res, const int rows, const int cols,
                         const int my_rank, const int comm_sz) {
    // 1. Создание топологии процессов
    // Определяем размеры решетки (p_row x p_col)
    int p_row = 1, p_col = comm_sz;
    for (int i = 1; i * i <= comm_sz; ++i) {
        if (comm_sz % i == 0) {
            p_row = i;
            p_col = comm_sz / i;
        }
    }

    // Создаем декартову решетку
    int dims[2] = {p_row, p_col};
    int periods[2] = {0, 0};
    int reorder = 1;
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);

    // Получаем ранг и координаты в новой решетке
    int grid_rank;
    int grid_coords[2];
    MPI_Comm_rank(grid_comm, &grid_rank);
    MPI_Cart_coords(grid_comm, grid_rank, 2, grid_coords);
    int my_row = grid_coords[0];
    int my_col = grid_coords[1];

    // 2. Расчет локальных размеров и смещений
    int base_rows = rows / p_row;
    int rem_rows = rows % p_row;
    int local_rows = base_rows + (my_row < rem_rows ? 1 : 0);
    int start_row = 0;
    for (int i = 0; i < my_row; ++i) {
        start_row += base_rows + (i < rem_rows ? 1 : 0);
    }

    int base_cols = cols / p_col;
    int rem_cols = cols % p_col;
    int local_cols = base_cols + (my_col < rem_cols ? 1 : 0);
    int start_col = 0;
    for (int j = 0; j < my_col; ++j) {
        start_col += base_cols + (j < rem_cols ? 1 : 0);
    }

    // Выделение памяти для локальных данных
    double *local_mat = calloc(local_rows * local_cols, sizeof(double));
    double *local_vec = calloc(local_cols, sizeof(double));

    // 3. Распределение матрицы от корневого процесса
    if (my_rank == kRoot) {
        for (int i = 0; i < p_row; ++i) {
            int block_rows = base_rows + (i < rem_rows ? 1 : 0);
            int block_start_row = 0;
            for (int r = 0; r < i; ++r) {
                block_start_row += base_rows + (r < rem_rows ? 1 : 0);
            }
            for (int j = 0; j < p_col; ++j) {
                // Получаем ранг процесса по его координатам, а не вычисляем вручную.
                int dest_coords[2] = {i, j};
                int dest_rank;
                MPI_Cart_rank(grid_comm, dest_coords, &dest_rank);

                if (dest_rank == kRoot) continue; // Пропускаем отправку самому себе

                int block_cols = base_cols + (j < rem_cols ? 1 : 0);
                int block_start_col = 0;
                for (int c = 0; c < j; ++c) {
                    block_start_col += base_cols + (c < rem_cols ? 1 : 0);
                }

                double *block_buf = malloc(block_rows * block_cols * sizeof(double));
                for (int r = 0; r < block_rows; ++r) {
                    memcpy(block_buf + r * block_cols, 
                           mat + (block_start_row + r) * cols + block_start_col, 
                           block_cols * sizeof(double));
                }
                MPI_Send(block_buf, block_rows * block_cols, MPI_DOUBLE, dest_rank, 0, grid_comm);
                free(block_buf);
            }
        }
        // Копируем блок для самого корневого процесса
        for (int r = 0; r < local_rows; ++r) {
            memcpy(local_mat + r * local_cols, 
                   mat + (start_row + r) * cols + start_col, 
                   local_cols * sizeof(double));
        }
    } else {
        MPI_Recv(local_mat, local_rows * local_cols, MPI_DOUBLE, kRoot, 0, grid_comm, MPI_STATUS_IGNORE);
    }

    // 4. Распределение вектора
    // Создаем коммуникаторы для каждого столбца
    MPI_Comm col_comm;
    MPI_Comm_split(grid_comm, my_col, my_row, &col_comm);
    
    // Лидеры столбцов (процессы в первой строке) получают свои части вектора от kRoot
    if (my_row == 0) {
        if (my_rank == kRoot) {
            for (int j = 0; j < p_col; ++j) {
                int block_cols = base_cols + (j < rem_cols ? 1 : 0);
                int block_start_col = 0;
                for (int c = 0; c < j; ++c) {
                    block_start_col += base_cols + (c < rem_cols ? 1 : 0);
                }
                
                // Получаем ранг процесса-лидера по его координатам.
                int dest_coords[2] = {0, j};
                int dest_rank;
                MPI_Cart_rank(grid_comm, dest_coords, &dest_rank);

                if (dest_rank == kRoot) {
                    memcpy(local_vec, vec + block_start_col, block_cols * sizeof(double));
                } else {
                    MPI_Send(vec + block_start_col, block_cols, MPI_DOUBLE, dest_rank, 1, grid_comm);
                }
            }
        } else {
            MPI_Recv(local_vec, local_cols, MPI_DOUBLE, kRoot, 1, grid_comm, MPI_STATUS_IGNORE);
        }
    }

    // Рассылка (Broadcast) сегмента вектора внутри каждого столбца
    // Лидер столбца (ранг 0 в col_comm) рассылает данные всем остальным.
    MPI_Bcast(local_vec, local_cols, MPI_DOUBLE, 0, col_comm);

    // 5. Локальное умножение матрицы на вектор
    double *local_res = calloc(local_rows, sizeof(double));
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < local_cols; ++j) {
            local_res[i] += local_mat[i * local_cols + j] * local_vec[j];
        }
    }

    // 6. Сборка результата
    // Создаем коммуникаторы для каждой строки
    MPI_Comm row_comm;
    MPI_Comm_split(grid_comm, my_row, my_col, &row_comm);

    // Редукция (суммирование) частичных результатов внутри каждой строки.
    // Результат оказывается у лидера строки (процесса с my_col == 0).
    double *row_res = NULL;
    if (my_col == 0) {
        row_res = calloc(local_rows, sizeof(double));
    }
    MPI_Reduce(local_res, row_res, local_rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    // Сбор итоговых результатов от лидеров строк к корневому процессу.
    // Эта операция должна происходить на коммуникаторе, объединяющем всех лидеров строк,
    // то есть на col_comm для процессов, где my_col == 0.
    if (my_col == 0) {
        if (my_rank == kRoot) {
            int *recvcounts = malloc(p_row * sizeof(int));
            int *displs = malloc(p_row * sizeof(int));
            displs[0] = 0;
            for (int i = 0; i < p_row; ++i) {
                recvcounts[i] = base_rows + (i < rem_rows ? 1 : 0);
                if (i > 0) {
                    displs[i] = displs[i-1] + recvcounts[i-1];
                }
            }
            // Корень (ранг 0 в col_comm первого столбца) собирает данные.
            MPI_Gatherv(row_res, local_rows, MPI_DOUBLE, res, recvcounts, displs, MPI_DOUBLE, 0, col_comm);
            free(recvcounts);
            free(displs);
        } else {
            // Остальные лидеры строк отправляют свои данные корню своего col_comm.
            MPI_Gatherv(row_res, local_rows, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, col_comm);
        }
        free(row_res);
    }
    
    // 7. Очистка ресурсов
    free(local_mat);
    free(local_vec);
    free(local_res);
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);

    // Замер времени
    double local_time = MPI_Wtime();
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, kRoot, MPI_COMM_WORLD);
    return max_time;
}

int main(int argc, char *argv[]) {

    int comm_sz;
    int my_rank;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    enum Mode { ROW, COL, BLOCK } mode = ROW;
    const int equal = 0;
    if (argc >= 2) {
        if (strcmp(argv[1], "row") == equal) {
            mode = ROW;
        } else if (strcmp(argv[1], "col") == equal) {
            mode = COL;
        } else if (strcmp(argv[1], "block") == equal) {
            mode = BLOCK;
        } else {
            if (my_rank == kRoot) {
                fprintf(stderr,
                        "Invalid argument: incorrect mode. Mod can only have following values: "
                        "row, col or block.\n");
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    } else {
        if (my_rank == kRoot) {
            fprintf(stderr,
                    "Invalid argument: mode is required. Mod can only have following values: row, "
                    "col or block.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int rows = rand() % 10, cols = rand() % 10;
    if (argc >= 3) {
        for (uint32_t i = 0; i < strlen(argv[2]); ++i) {
            if (!isdigit(argv[2][i])) {
                if (my_rank == kRoot) {
                    fprintf(stderr,
                            "Invalid argument: expect positive integer number - dimension of the matrix or rows and cols.\n");
                }
                MPI_Finalize();
                return EXIT_FAILURE;
            }
        }

        char *endptr = NULL;
        rows = cols = atoi(argv[2]);
        if (rows == 0) {
            if (my_rank == kRoot) {
                fprintf(stderr,
                        "Invalid argument: expect positive integer number - dimension of the matrix or rows and cols.\n");
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }
    if (argc >= 4) {
        for (uint32_t i = 0; i < strlen(argv[3]); ++i) {
            if (!isdigit(argv[3][i])) {
                if (my_rank == kRoot) {
                    fprintf(stderr,
                            "Invalid argument: expect positive integer number - dimension of the matrix or rows and cols.\n");
                }
                MPI_Finalize();
                return EXIT_FAILURE;
            }
        }

        char *endptr = NULL;
        cols = atoi(argv[3]);
        if (rows == 0) {
            if (my_rank == kRoot) {
                fprintf(stderr,
                        "Invalid argument: expect positive integer number - dimension of the matrix or rows and cols.\n");
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    double *vec = NULL, *mat = NULL;
    MPI_Bcast(&rows, 1, MPI_INT, kRoot, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, kRoot, MPI_COMM_WORLD);
    double *res = calloc(rows, sizeof(double));
    if (my_rank == kRoot) {
        vec = calloc(cols, sizeof(double));
        mat = calloc(rows * cols, sizeof(double));
        InitVector(vec, cols);
        InitVector(mat, rows * cols);
    }

    double total_time = 0;
    switch (mode) {
        case ROW:
            total_time = MatVecInRowMode(mat, vec, res, rows, cols, my_rank, comm_sz);
            break;
        case COL:
            total_time = MatVecInColMode(mat, vec, res, rows, cols, my_rank, comm_sz);
            break;
        case BLOCK:
            total_time = MatVecInBlockMode(mat, vec, res, rows, cols, my_rank, comm_sz);
            break;
    }

    if (my_rank == kRoot) {
#ifdef DEBUG
        PrintMat(mat, rows, cols);
        PrintVec(vec, cols);
        PrintVec(res, rows);
#endif
        printf("Total time = %f seconds\n", total_time);
    }

    MPI_Finalize();

    free(res);
    if (my_rank == kRoot) {
        free(vec);
        free(mat);
    }

    return 0;
}
