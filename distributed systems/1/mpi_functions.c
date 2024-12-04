#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mpi_functions.h"

#define N_ROWS 5
#define N_COLS 5



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


/**
 * @brief Performs reduction operation to find the maximum value.
 *
 * This function exchange_datas with neighboring processes to find the 
 * maximum value across the grid of processes.
 */
void find_max(int *data, int *best_rank, MPI_Comm comm, int coord[2]) {
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

            if (col > 0) {
                exchange_data(directions[0][0], directions[0][1], &other_data, &other_rank, data, best_rank, 0, comm);
            }
            if (col < 4) {
                exchange_data(directions[1][0], directions[1][1], &other_data, &other_rank, data, best_rank, send, comm);
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
