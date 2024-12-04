#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#define N_ROW 5
#define N_COL 5


int size = N_ROW * N_COL;
int rank;


void exchange_data(int neighbour_row, int neighbour_col, 
                 int *recv_data, int *recv_rank, 
                 int *data, int *best_rank, 
                 int is_sending, MPI_Comm comm) {
    
    int neighbour_coords[2] = {neighbour_row, neighbour_col};
    int neighbour_rank;
    MPI_Cart_rank(comm, neighbour_coords, &neighbour_rank);

    if (is_sending) {
        MPI_Send(best_rank, 1, MPI_INT, neighbour_rank, 0, comm);
        MPI_Send(data, 1, MPI_INT, neighbour_rank, 0, comm);
    } else {
        MPI_Status status;
        MPI_Recv(recv_rank, 1, MPI_INT, neighbour_rank, 0, comm, &status);
        MPI_Recv(recv_data, 1, MPI_INT, neighbour_rank, 0, comm, &status);
        
        if (*recv_data > *data) {
            *data = *recv_data;
            *best_rank = *recv_rank;
        }
    }
}


void find_max(
    int *data,
    int *best_rank,
    MPI_Comm comm,
    int coord[2]
) {
    int is_sending = 1;
    int neighbour_rank;
    int neighbour_data;
    int row = coord[0];
    int col = coord[1];

    // Определяем направление для отправки и получения
    int directions[2][2];

    // Обработка по оси X
    switch (row) {
        case 0:
        case 1:
            if (row > 0) {
                directions[0][0] = row - 1; directions[0][1] = col; // вверх
                exchange_data(directions[0][0], directions[0][1], &neighbour_data, &neighbour_rank, data, best_rank, 0, comm);
            }
            directions[1][0] = row + 1; directions[1][1] = col; // вниз
            exchange_data(directions[1][0], directions[1][1], &neighbour_data, &neighbour_rank, data, best_rank, is_sending, comm);
            break;

        case 3:
        case 4:
            if (row < 4) {
                directions[1][0] = row + 1; directions[1][1] = col; // вниз
                exchange_data(directions[1][0], directions[1][1], &neighbour_data, &neighbour_rank, data, best_rank, 0, comm);
            }
            directions[0][0] = row - 1; directions[0][1] = col; // вверх
            exchange_data(directions[0][0], directions[0][1], &neighbour_data, &neighbour_rank, data, best_rank, is_sending, comm);
            break;

        default:
            // Обработка по оси Y
            directions[0][0] = row - 1; directions[0][1] = col; // вверх
            directions[1][0] = row + 1; directions[1][1] = col; // вниз
            for (int i = 0; i < 2; i++) {
                exchange_data(directions[i][0], directions[i][1], &neighbour_data, &neighbour_rank, data, best_rank, 0, comm);
            }

            // Обработка по оси Y
            directions[0][0] = row; directions[0][1] = col - 1; // влево
            directions[1][0] = row; directions[1][1] = col + 1; // вправо

            switch (col) {
                case 0:
                case 1:
                    if (col > 0) {
                        exchange_data(directions[0][0], directions[0][1], &neighbour_data, &neighbour_rank, data, best_rank, 0, comm);
                    }
                    exchange_data(directions[1][0], directions[1][1], &neighbour_data, &neighbour_rank, data, best_rank, is_sending, comm);
                    break;

                case 3:
                case 4:
                    if (col < 4) {
                        exchange_data(directions[1][0], directions[1][1], &neighbour_data, &neighbour_rank, data, best_rank, 0, comm);
                    }
                    exchange_data(directions[0][0], directions[0][1], &neighbour_data, &neighbour_rank, data, best_rank, is_sending, comm);
                    break;

                default:
                    for (int i = 0; i < 2; i++) {
                        exchange_data(directions[i][0], directions[i][1], &neighbour_data, &neighbour_rank, data, best_rank, 0, comm);
                    }
                    break;
            }
            break;
    }
}


void send_to_all(int *data, int *best_rank, MPI_Comm comm, int coord[2]) {
    int is_sending = 1;
    int neighbour_rank;
    int neighbour_data;

    switch (coord[0]) {
        case 0:
        case 1:
            // Передача вниз
            exchange_data(coord[0] + 1, coord[1], &neighbour_data, &neighbour_rank, data, best_rank, !is_sending, comm);
            if (coord[0] > 0) {
                // Передача вверх
                exchange_data(coord[0] - 1, coord[1], &neighbour_data, &neighbour_rank, data, best_rank, is_sending, comm);
            }
            break;

        case 3:
        case 4:
            // Передача вверх
            exchange_data(coord[0] - 1, coord[1], &neighbour_data, &neighbour_rank, data, best_rank, !is_sending, comm);
            if (coord[0] < 4) {
                // Передача вниз
                exchange_data(coord[0] + 1, coord[1], &neighbour_data, &neighbour_rank, data, best_rank, is_sending, comm);
            }
            break;

        default:
            // Обработка по горизонтали (осе Y)
            switch (coord[1]) {
                case 0:
                case 1:
                    exchange_data(coord[0], coord[1] + 1, &neighbour_data, &neighbour_rank, data, best_rank, !is_sending, comm);
                    if (coord[1] > 0) {
                        exchange_data(coord[0], coord[1] - 1, &neighbour_data, &neighbour_rank, data, best_rank, is_sending, comm);
                    }
                    break;

                case 3:
                case 4:
                    exchange_data(coord[0], coord[1] - 1, &neighbour_data, &neighbour_rank, data, best_rank, !is_sending, comm);
                    if (coord[1] < 4) {
                        exchange_data(coord[0], coord[1] + 1, &neighbour_data, &neighbour_rank, data, best_rank, is_sending, comm);
                    }
                    break;

                default:
                    // Передача по обеим осям
                    exchange_data(coord[0], coord[1] - 1, &neighbour_data, &neighbour_rank, data, best_rank, is_sending, comm);
                    exchange_data(coord[0], coord[1] + 1, &neighbour_data, &neighbour_rank, data, best_rank, is_sending, comm);
                    break;
            }
            // Передача по вертикали
            exchange_data(coord[0] - 1, coord[1], &neighbour_data, &neighbour_rank, data, best_rank, is_sending, comm);
            exchange_data(coord[0] + 1, coord[1], &neighbour_data, &neighbour_rank, data, best_rank, is_sending, comm);
            break;
    }
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm comm;
    int dims[2] = {N_ROW, N_COL};
    int periods[2] = {0};
    int coords[2];

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm);
    MPI_Cart_coords(comm, rank, 2, coords);

    srand(rank);
    int data = rand() % 1000000;
    int best_rank = rank;

    if (rank == 0) { printf("Generated data:\n"); }
    MPI_Barrier(comm);

    for (int i = 0; i < size; i++) {
        if (i == rank) {
            printf("rank: %d \tcoords: %d, %d\tdata: %d\n", rank, coords[0], coords[1], data);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    find_max(&data, &best_rank, comm, coords);
    MPI_Barrier(comm);
    if (coords[0] == 2 && coords[1] == 2) { printf("\nMax data value: %d\nMax data rank: %d\n", data, best_rank); }

    send_to_all(&data, &best_rank, comm, coords);
    int best_coords[2];
    MPI_Cart_coords(comm, best_rank, 2, best_coords);
    if (rank == 0) { printf("\nData after send_to_all:\n"); }
    MPI_Barrier(comm);

    for (int i = 0; i < size; i++) {
        if (i == rank) {
            printf("rank: %d \tbest rank: %d\tbest coords: %d, %d\tdata: %d\n",
                   rank, best_rank, best_coords[0], best_coords[1], data);
            fflush(stdout);
        }
        MPI_Barrier(comm);
    }

    MPI_Finalize();
    return 0;
}
