#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void printMatrix(int *matrix, int nrows, int ncols) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            printf("%d\t", matrix[i * ncols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void multiplyCannon(int *blockA, int *blockB, int *blockC, int n){
    for (int i=0; i < n; i++){
        for (int j=0;j < n; j++){
            for (int k=0; k < n; k++){
                blockC[i * n + j] += blockA[i * n + k] * blockB[k * n + j];
            }
        }
    }
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int comm_sz, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (argc != 2){
        if (my_rank == 0) {
            printf("Количество аргументов должно быть 2 - количество процессов(comm_size) и размерность квадратной матрицы (n)\n");
        }
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    int sqrt_comm_sz = (int)sqrt(comm_sz);
    if (sqrt_comm_sz * sqrt_comm_sz != comm_sz) {
        if (my_rank == 0) {
            printf("Число процессов (comm_size) должен быть полным квадратом числа\n");
        }
        MPI_Finalize();
        return 1;
    }
    if (n % sqrt_comm_sz != 0) {
        if (my_rank == 0) printf("Размерность матрицы (n) должна делиться на sqrt(comm_size)\n");
        MPI_Finalize();
        return 1;
    }

    int local_n = n / sqrt_comm_sz;

    MPI_Comm MPI_grid_comm;
    int dims[2] = {sqrt_comm_sz, sqrt_comm_sz};
    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &MPI_grid_comm);

    int grid_rank, coords[2];
    MPI_Comm_rank(MPI_grid_comm, &grid_rank);
    MPI_Cart_coords(MPI_grid_comm, grid_rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    int left, right, up, down;
    MPI_Cart_shift(MPI_grid_comm, 1, 1, &left, &right);
    MPI_Cart_shift(MPI_grid_comm, 0, 1, &up, &down);

    int *A = NULL, *B = NULL, *C = NULL;
    if (grid_rank == 0) {
        A = malloc(n*n*sizeof(int));
        B = malloc(n*n*sizeof(int));
        C = calloc(n*n, sizeof(int));
        srand(42);
        for (int i = 0; i < n*n; i++) {
            A[i] = rand() % 10;
            B[i] = rand() % 10;
        }
        if (n < 16){
            printf("Matrix A:\n");
            printMatrix(A, n, n);
            printf("Matrix B:\n");
            printMatrix(B, n, n);
        }
    }

    int *blockA = calloc(local_n*local_n, sizeof(int));
    int *blockB = calloc(local_n*local_n, sizeof(int));
    int *blockC = calloc(local_n*local_n, sizeof(int));

    MPI_Datatype block_type;
    MPI_Type_vector(local_n, local_n, n, MPI_INT, &block_type);
    MPI_Type_commit(&block_type);

    if (my_rank == 0) {
        for (int proc=0;proc<comm_sz; proc++){
            int proc_coords[2];
            MPI_Cart_coords(MPI_grid_comm, proc, 2, proc_coords);
            int row_offset = proc_coords[0] * local_n;
            int col_offset = proc_coords[1] * local_n;
            int offset = row_offset * n + col_offset;

            if (proc == 0) {
                for (int i = 0; i < local_n; i++) {
                    for (int j = 0; j < local_n; j++) {
                        blockA[i * local_n + j] = A[(row_offset + i) * n + col_offset + j];
                        blockB[i * local_n + j] = B[(row_offset + i) * n + col_offset + j];
                    }
                }
            } else {
                MPI_Send(A + offset, 1, block_type, proc, 0, MPI_grid_comm);
                MPI_Send(B + offset, 1, block_type, proc, 1, MPI_grid_comm);
            }
        }
    } else{
        MPI_Recv(blockA, local_n * local_n, MPI_INT, 0, 0,
                 MPI_grid_comm, MPI_STATUS_IGNORE);
        MPI_Recv(blockB, local_n * local_n, MPI_INT, 0, 1,
                 MPI_grid_comm, MPI_STATUS_IGNORE);
    }

    MPI_Status status;
    for (int i = 0; i < my_row; i++) {
        MPI_Sendrecv_replace(blockA, local_n*local_n, MPI_INT, left, 0, right, 0, MPI_grid_comm, &status);
    }
    for (int i = 0; i < my_col; i++) {
        MPI_Sendrecv_replace(blockB, local_n*local_n, MPI_INT, up, 0, down, 0, MPI_grid_comm, &status);
    }

    MPI_Barrier(MPI_grid_comm);
    double start_time = MPI_Wtime();

    for (int i = 0; i < sqrt_comm_sz; i++) {
        multiplyCannon(blockA, blockB, blockC, local_n);
        MPI_Sendrecv_replace(blockA, local_n*local_n, MPI_INT, left, 0, right, 0, MPI_grid_comm, &status);
        MPI_Sendrecv_replace(blockB, local_n*local_n, MPI_INT, up, 0, down, 0, MPI_grid_comm, &status);
    }

    int *tmp_buffer = NULL;
    if (grid_rank == 0) {
        tmp_buffer = malloc(local_n * local_n * comm_sz * sizeof(int));
    }

    MPI_Gather(blockC, local_n * local_n, MPI_INT,
               tmp_buffer, local_n * local_n, MPI_INT,
               0, MPI_grid_comm);

    if (my_rank == 0) {
        for (int proc = 0; proc < comm_sz; proc++) {
            int proc_coords[2];
            MPI_Cart_coords(MPI_grid_comm, proc, 2, proc_coords);
            int row_offset = proc_coords[0] * local_n;
            int col_offset = proc_coords[1] * local_n;

            for (int i = 0; i < local_n; i++) {
                for (int j = 0; j < local_n; j++) {
                    int src_index = proc * local_n * local_n + i * local_n + j;
                    int dest_index = (row_offset + i) * n + (col_offset + j);
                    C[dest_index] = tmp_buffer[src_index];
                }
            }
        }
    }

    MPI_Barrier(MPI_grid_comm);
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;

    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0 && n < 16) {
        printf("Matrix C (Result):\n");
        printMatrix(C, n, n);
    }

    if (my_rank == 0) {
        printf("\n--- Результаты замеров ---\n");
        printf("Размер матрицы n: %d\n", n);
        printf("Количество процессов(comm_size): %d (%dx%d)\n", comm_sz, sqrt_comm_sz, sqrt_comm_sz);
        printf("Максимальное время выполнения: %f секунд\n", max_time);
    }

    MPI_Type_free(&block_type);
    free(blockA);
    free(blockB);
    free(blockC);

    if (grid_rank == 0) {
        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();
    return 0;
}
