#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mpi_functions.h"

#define N_ROW 5
#define N_COL 5


void initialize_mpi(int *rank, MPI_Comm *comm) {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);

    int dims[2] = {N_ROW, N_COL};
    int periods[2] = {0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, comm);
}

void generate_data(int rank, int *data) {
    *data = rand() % 1000000;
}

void print_generated_data(int rank, int data, int *coords) {
    if (rank == 0) { 
        printf("GENERATED DATA:\n");
        printf("───────────────────────────────────────────────\n");
        printf(" Rank │ Coords   │ Data                        \n");
        printf("───────────────────────────────────────────────\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < N_ROW * N_COL; i++) {
        if (i == rank) {
            printf("│ %4d │ (%2d,%2d) │ %d                     \n", 
                   rank, coords[0], coords[1], data);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("───────────────────────────────────────────────\n");
    }
}


void print_broadcast_data(int rank, int best_rank, int *best_coords, int data) {
    if (rank == 0) { 
        printf("\nRESULT DATA:\n");
        printf("───────────────────────────────────────────────\n");
        printf(" Rank │ Best Rank │ Coords      │ Value        \n");
        printf("───────────────────────────────────────────────\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < N_ROW * N_COL; i++) {
        if (i == rank) {
            printf(" %4d │ %9d │ (%2d,%2d)     │ %d           \n", 
                   rank, best_rank, best_coords[0], best_coords[1], data);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("───────────────────────────────────────────────\n");
    }
}

int main(int argc, char *argv[]) {
    int rank;
    MPI_Comm comm;

    initialize_mpi(&rank, &comm);

    srand(rank);
    int data;
    generate_data(rank, &data);

    int coords[2];
    MPI_Cart_coords(comm, rank, 2, coords);
    print_generated_data(rank, data, coords);

    int best_rank = rank;
    find_max(&data, &best_rank, comm, coords);
    MPI_Barrier(comm);

    send_to_all(&data, &best_rank, comm, coords);
    MPI_Barrier(comm);

    int best_coords[2];
    MPI_Cart_coords(comm, best_rank, 2, best_coords);
    print_broadcast_data(rank, best_rank, best_coords, data);

    MPI_Finalize();
    return 0;
}
