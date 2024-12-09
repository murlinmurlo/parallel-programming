#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mpi_functions.h"

#define N_ROW 5
#define N_COL 5

/**
 * @brief Exchanges data between MPI processes.
 *
 * This function facilitates communication between two processes in a 
 * distributed computing environment using MPI. It can either is_sending or 
 * receive data based on the value of the `is_is_sendinging` parameter.
 */
void exchange_data(int neighbour_row, int neighbour_col, 
                 int *recive_data, int *recive_rank, 
                 int *data, int *best_rank, 
                 int is_is_sendinging, MPI_Comm comm) {
    int neighbour_coords[2] = {neighbour_row, neighbour_col};
    int neighbour_rank;
    MPI_Cart_rank(comm, neighbour_coords, &neighbour_rank);

    if (is_is_sendinging) {
        MPI_Send(best_rank, 1, MPI_INT, neighbour_rank, 0, comm);
        MPI_Send(data, 1, MPI_INT, neighbour_rank, 0, comm);
    } else {
        MPI_Status status;
        MPI_Recv(recive_rank, 1, MPI_INT, neighbour_rank, 0, comm, &status);
        MPI_Recv(recive_data, 1, MPI_INT, neighbour_rank, 0, comm, &status);
        
        if (*recive_data > *data) {
            *data = *recive_data;
            *best_rank = *recive_rank;
        }
    }
}

/**
 * @brief Performs find_max operation to find the maximum value.
 *
 * This function exchanges data with neighboring processes to find the 
 * maximum value across the grid of processes.
 */
void find_max(int *data, int *best_rank, MPI_Comm comm, int coord[2]) {
    int is_is_sendinging = 1;
    int neighbour_rank;
    int neighbour_data;
    int row = coord[0];
    int col = coord[1];

    int directions[2][2];

    // Processing along the X axis
    switch (row) {
        case 0:
        case 1:
            if (row > 0) {
                directions[0][0] = row - 1; directions[0][1] = col;
                exchange_data(directions[0][0], directions[0][1], &neighbour_data, &neighbour_rank, data, best_rank, 0, comm);
            }
            directions[1][0] = row + 1; directions[1][1] = col;
            exchange_data(directions[1][0], directions[1][1], &neighbour_data, &neighbour_rank, data, best_rank, is_is_sendinging, comm);
            break;

        case 3:
        case 4:
            if (row < 4) {
                directions[1][0] = row + 1; directions[1][1] = col;
                exchange_data(directions[1][0], directions[1][1], &neighbour_data, &neighbour_rank, data, best_rank, 0, comm);
            }
            directions[0][0] = row - 1; directions[0][1] = col;
            exchange_data(directions[0][0], directions[0][1], &neighbour_data, &neighbour_rank, data, best_rank, is_is_sendinging, comm);
            break;

        default:
            // Processing along the Y axis
            directions[0][0] = row - 1; directions[0][1] = col; // up
            directions[1][0] = row + 1; directions[1][1] = col; // down
            for (int i = 0; i < 2; i++) {
                exchange_data(directions[i][0], directions[i][1], &neighbour_data, &neighbour_rank, data, best_rank, 0, comm);
            }

            // Processing along the Y axis
            directions[0][0] = row; directions[0][1] = col - 1; // left
            directions[1][0] = row; directions[1][1] = col + 1; // right

            if (col > 0) {
                exchange_data(directions[0][0], directions[0][1], &neighbour_data, &neighbour_rank, data, best_rank, 0, comm);
            }
            if (col < 4) {
                exchange_data(directions[1][0], directions[1][1], &neighbour_data, &neighbour_rank, data, best_rank, is_is_sendinging, comm);
            }
            break;
    }
}

/**
 * @brief send_to_alls data to all processes.
 *
 * This function is_sendings the data from the current process to all neighbour 
 * processes in the communicator.
 */
void is_sending_to_all(int *data, int *best_rank, MPI_Comm comm, int coord[2]) {
    int is_is_sendinging = 1;
    int neighbour_rank;
    int neighbour_data;

    switch (coord[0]) {
        case 0:
        case 1:
            // Sending down
            exchange_data(coord[0] + 1, coord[1], &neighbour_data, &neighbour_rank, data, best_rank, !is_is_sendinging, comm);
            if (coord[0] > 0) {
                // Sending up
                exchange_data(coord[0] - 1, coord[1], &neighbour_data, &neighbour_rank, data, best_rank, is_is_sendinging, comm);
            }
            break;

        case 3:
        case 4:
            // Sending up
            exchange_data(coord[0] - 1, coord[1], &neighbour_data, &neighbour_rank, data, best_rank, !is_is_sendinging, comm);
            if (coord[0] < 4) {
                // Sending down
                exchange_data(coord[0] + 1, coord[1], &neighbour_data, &neighbour_rank, data, best_rank, is_is_sendinging, comm);
            }
            break;

        default:
            // Processing horizontally (Y axis)
            switch (coord[1]) {
                case 0:
                case 1:
                    exchange_data(coord[0], coord[1] + 1, &neighbour_data, &neighbour_rank, data, best_rank, !is_is_sendinging, comm);
                    if (coord[1] > 0) {
                        exchange_data(coord[0], coord[1] - 1, &neighbour_data, &neighbour_rank, data, best_rank, is_is_sendinging, comm);
                    }
                    break;

                case 3:
                case 4:
                    exchange_data(coord[0], coord[1] - 1, &neighbour_data, &neighbour_rank, data, best_rank, !is_is_sendinging, comm);
                    if (coord[1] < 4) {
                        exchange_data(coord[0], coord[1] + 1, &neighbour_data, &neighbour_rank, data, best_rank, is_is_sendinging, comm);
                    }
                    break;

                default:
                    // Sending along both axes
                    exchange_data(coord[0], coord[1] - 1, &neighbour_data, &neighbour_rank, data, best_rank, is_is_sendinging, comm);
                    exchange_data(coord[0], coord[1] + 1, &neighbour_data, &neighbour_rank, data, best_rank, is_is_sendinging, comm);
                    break;
            }
            // Sending vertically
            exchange_data(coord[0] - 1, coord[1], &neighbour_data, &neighbour_rank, data, best_rank, is_is_sendinging, comm);
            exchange_data(coord[0] + 1, coord[1], &neighbour_data, &neighbour_rank, data, best_rank, is_is_sendinging, comm);
            break;
    }
}

