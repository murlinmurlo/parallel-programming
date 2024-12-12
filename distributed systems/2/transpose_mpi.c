#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <signal.h>

#define N 10
#define NPROC 64
#define READ_MATRIX_FROM_FILE 0


MPI_Comm main_comm;

void
prnmtrx1d(int n, int a[n * n])
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            printf("%d ", a[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void
prnmtrx2d(int n, int a[n][n])
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void
init_matrix(int n, int a[n * n])
{
    for (size_t i = 0; i < n * n; i++)
    {
        a[i] = i;
    }
}

int
transpose(int n, int resMpi[n][n], int rank, int nProc)
{   
    static int count = 0;
    count++;

    size_t i = 0, j, idx;
    int rowMpi[n];
    int a[n * n];
    MPI_Request request;
    if (rank == 0) {
        FILE *input = fopen("checkpoint.txt", "r");
        for (int k = 0; k < n * n; k++) {
            fscanf(input, "%d", &a[k]);
        }
        fclose(input);
    }
    MPI_Barrier(main_comm);


    int err = MPI_Bcast(a, n * n, MPI_INT, 0, main_comm);
    if (err != MPI_SUCCESS) {
        printf("Proccess %d failed on launch %d\n", rank, count); fflush(stdout);
        return -1;
    }



    while (rank + i < n) {
        for (j = 0; j < n; j++)
        {
            idx = j * n + rank + i;
            if (idx < n * n) {
                if (!rank) {
                    resMpi[i][j] = a[idx];
                } else {
                    rowMpi[j] = a[idx];
                }
            }
        }
        if (rank) {
            err = MPI_Isend(rowMpi, n, MPI_INT, 0, i, main_comm, &request);
            if (err != MPI_SUCCESS) {
                printf("Proccess %d failed on launch %d\n", rank, count); fflush(stdout);
                return -1;
            }
            MPI_Barrier(main_comm);

            err = MPI_Wait(&request, MPI_STATUS_IGNORE);
            if (err != MPI_SUCCESS) {
                printf("Proccess %d failed on launch %d\n", rank, count); fflush(stdout);
                return -1;
            }
            MPI_Barrier(main_comm);
        }
        i += nProc - 1;
    }

    MPI_Barrier(main_comm);



    if (!rank) {
        printf("\n");
        for (int r = 1; r < nProc; r++) {
            int proc = 0;
            while (r + proc < N) {
                err = MPI_Irecv(resMpi[r + proc], n, MPI_INT, r, proc, main_comm, &request);
                if (err != MPI_SUCCESS) {
                    printf("Proccess %d failed on launch %d\n", rank, count); fflush(stdout);
                    return -1;
                }

                err = MPI_Wait(&request, MPI_STATUS_IGNORE);
                if (err != MPI_SUCCESS) {
                    printf("Proccess %d failed on launch %d\n", rank, count); fflush(stdout);
                    return -1;
                }

                proc += nProc;
            }
        }
    }
    return 0;
}

void solver(int n, int resMpi[n][n])
{   
    int rank, nProc;
    MPI_Comm_size(main_comm, &nProc);
    MPI_Comm_rank(main_comm, &rank);
    int res = transpose(n, resMpi, rank, nProc);
    printf("%d\n", res);
    if (res < 0) {
        MPIX_Comm_shrink(main_comm, &main_comm);
        solver(n, resMpi);
        return;
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    main_comm = MPI_COMM_WORLD;
    MPI_Comm_set_errhandler(main_comm, MPI_ERRORS_RETURN);
    MPI_Comm_rank(main_comm, &rank);

    if (!rank) {
        printf("running on %d processes\n", NPROC);
        printf("<OUTPUT>\n");
    }

    MPI_Request request;
    int a[N * N];
    int resMpi[N][N];
    if (!rank) {
        init_matrix(N, a);
        FILE *output = fopen("checkpoint.txt", "w");
        for (int k = 0; k < N * N; k++) {
            fprintf(output, "%d ", a[k]);
        }
        fclose(output);
    }
    
    MPI_Barrier(main_comm);
    solver(N, resMpi);
    MPI_Barrier(main_comm);

    if (!rank) {
        prnmtrx2d(N, resMpi);
        printf("<OUTPUT>\n");
    }

    MPI_Finalize();
    return 0;
}