#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#define N_ROWS 5
#define N_COLS 5


int size = N_ROWS * N_COLS;
int rank;


/**
 * @brief Exchanges data between MPI processes.
 *
 * This function facilitates communication between two processes in a 
 * distributed computing environment using MPI. It can either send or 
 * receive data based on the value of the `send` parameter.
 */
void exchange_data(int other_row, int other_col, 
                 int *recv_data, int *recv_rank, 
                 int *data, int *best_rank, 
                 int send, MPI_Comm comm) {
    
    int other_coords[2] = {other_row, other_col};
    int other_rank;
    MPI_Cart_rank(comm, other_coords, &other_rank);

    if (send) {
        MPI_Send(best_rank, 1, MPI_INT, other_rank, 0, comm);
        MPI_Send(data, 1, MPI_INT, other_rank, 0, comm);
    } else {
        MPI_Status status;
        MPI_Recv(recv_rank, 1, MPI_INT, other_rank, 0, comm, &status);
        MPI_Recv(recv_data, 1, MPI_INT, other_rank, 0, comm, &status);
        
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
    int send = 1;
    int other_rank;
    int other_data;
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
                exchange_data(directions[0][0], directions[0][1], &other_data, &other_rank, data, best_rank, 0, comm);
            }
            directions[1][0] = row + 1; directions[1][1] = col; // вниз
            exchange_data(directions[1][0], directions[1][1], &other_data, &other_rank, data, best_rank, send, comm);
            break;

        case 3:
        case 4:
            if (row < 4) {
                directions[1][0] = row + 1; directions[1][1] = col; // вниз
                exchange_data(directions[1][0], directions[1][1], &other_data, &other_rank, data, best_rank, 0, comm);
            }
            directions[0][0] = row - 1; directions[0][1] = col; // вверх
            exchange_data(directions[0][0], directions[0][1], &other_data, &other_rank, data, best_rank, send, comm);
            break;

        default:
            // Обработка по оси Y
            directions[0][0] = row - 1; directions[0][1] = col; // вверх
            directions[1][0] = row + 1; directions[1][1] = col; // вниз
            for (int i = 0; i < 2; i++) {
                exchange_data(directions[i][0], directions[i][1], &other_data, &other_rank, data, best_rank, 0, comm);
            }

            // Обработка по оси Y
            directions[0][0] = row; directions[0][1] = col - 1; // влево
            directions[1][0] = row; directions[1][1] = col + 1; // вправо

            switch (col) {
                case 0:
                case 1:
                    if (col > 0) {
                        exchange_data(directions[0][0], directions[0][1], &other_data, &other_rank, data, best_rank, 0, comm);
                    }
                    exchange_data(directions[1][0], directions[1][1], &other_data, &other_rank, data, best_rank, send, comm);
                    break;

                case 3:
                case 4:
                    if (col < 4) {
                        exchange_data(directions[1][0], directions[1][1], &other_data, &other_rank, data, best_rank, 0, comm);
                    }
                    exchange_data(directions[0][0], directions[0][1], &other_data, &other_rank, data, best_rank, send, comm);
                    break;

                default:
                    for (int i = 0; i < 2; i++) {
                        exchange_data(directions[i][0], directions[i][1], &other_data, &other_rank, data, best_rank, 0, comm);
                    }
                    break;
            }
            break;
    }
}


/**
 * @brief Broadcasts data to all processes.
 *
 * This function sends the data from the current process to all other 
 * processes in the communicator.
 */
void send_to_all(int *data, int *best_rank, MPI_Comm comm, int coord[2]) {
    int send = 1;
    int other_rank;
    int other_data;

    switch (coord[0]) {
        case 0:
        case 1:
            // Передача вниз
            exchange_data(coord[0] + 1, coord[1], &other_data, &other_rank, data, best_rank, !send, comm);
            if (coord[0] > 0) {
                // Передача вверх
                exchange_data(coord[0] - 1, coord[1], &other_data, &other_rank, data, best_rank, send, comm);
            }
            break;

        case 3:
        case 4:
            // Передача вверх
            exchange_data(coord[0] - 1, coord[1], &other_data, &other_rank, data, best_rank, !send, comm);
            if (coord[0] < 4) {
                // Передача вниз
                exchange_data(coord[0] + 1, coord[1], &other_data, &other_rank, data, best_rank, send, comm);
            }
            break;

        default:
            // Обработка по горизонтали (осе Y)
            switch (coord[1]) {
                case 0:
                case 1:
                    exchange_data(coord[0], coord[1] + 1, &other_data, &other_rank, data, best_rank, !send, comm);
                    if (coord[1] > 0) {
                        exchange_data(coord[0], coord[1] - 1, &other_data, &other_rank, data, best_rank, send, comm);
                    }
                    break;

                case 3:
                case 4:
                    exchange_data(coord[0], coord[1] - 1, &other_data, &other_rank, data, best_rank, !send, comm);
                    if (coord[1] < 4) {
                        exchange_data(coord[0], coord[1] + 1, &other_data, &other_rank, data, best_rank, send, comm);
                    }
                    break;

                default:
                    // Передача по обеим осям
                    exchange_data(coord[0], coord[1] - 1, &other_data, &other_rank, data, best_rank, send, comm);
                    exchange_data(coord[0], coord[1] + 1, &other_data, &other_rank, data, best_rank, send, comm);
                    break;
            }
            // Передача по вертикали
            exchange_data(coord[0] - 1, coord[1], &other_data, &other_rank, data, best_rank, send, comm);
            exchange_data(coord[0] + 1, coord[1], &other_data, &other_rank, data, best_rank, send, comm);
            break;
    }
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm comm;
    int dims[2] = {N_ROWS, N_COLS};
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
