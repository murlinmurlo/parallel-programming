#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mpi_functions.h"

void exchange_data(int neighbour_row, int neighbour_col, 
                   int *recv_data, int *recv_rank, 
                   int *data, int *best_rank, 
                   int is_sending, MPI_Comm comm) {
    
    int neighbour_coords[2] = {neighbour_row, neighbour_col};
    int neighbour_rank;
    MPI_Cart_rank(comm, neighbour_coords, &neighbour_rank);

    int message[2];
    if (is_sending) {
        message[0] = *best_rank;
        message[1] = *data;
        MPI_Send(message, 2, MPI_INT, neighbour_rank, 0, comm);
    } else {
        MPI_Status status;
        MPI_Recv(message, 2, MPI_INT, neighbour_rank, 0, comm, &status);
        *recv_rank = message[0];
        *recv_data = message[1];

        if (*recv_data > *data) {
            *data = *recv_data;
            *best_rank = *recv_rank;
        }
    }
}

void find_max(int *data, int *best_rank, MPI_Comm comm, int coord[2]) {
    int is_sending = 1;
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
                directions[0][0] = row - 1; directions[0][1] = col; // up
                exchange_data(directions[0][0], directions[0][1], &neighbour_data, &neighbour_rank, data, best_rank, 0, comm);
            }
            directions[1][0] = row + 1; directions[1][1] = col; // down
            exchange_data(directions[1][0], directions[1][1], &neighbour_data, &neighbour_rank, data, best_rank, is_sending, comm);
            break;

        case 3:
        case 4:
            if (row < 4) {
                directions[1][0] = row + 1; directions[1][1] = col; // down
                exchange_data(directions[1][0], directions[1][1], &neighbour_data, &neighbour_rank, data, best_rank, 0, comm);
            }
            directions[0][0] = row - 1; directions[0][1] = col; // up
            exchange_data(directions[0][0], directions[0][1], &neighbour_data, &neighbour_rank, data, best_rank, is_sending, comm);
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
    int other_rank;
    int other_data;

    switch (coord[0]) {
        case 0:
        case 1:
            // Sending down
            exchange_data(coord[0] + 1, coord[1], &other_data, &other_rank, data, best_rank, !is_sending, comm);
            if (coord[0] > 0) {
                // Sending up
                exchange_data(coord[0] - 1, coord[1], &other_data, &other_rank, data, best_rank, is_sending, comm);
            }
            break;

        case 3:
        case 4:
            // Sending up
            exchange_data(coord[0] - 1, coord[1], &other_data, &other_rank, data, best_rank, !is_sending, comm);
            if (coord[0] < 4) {
                // Sending down
                exchange_data(coord[0] + 1, coord[1], &other_data, &other_rank, data, best_rank, is_sending, comm);
            }
            break;

        default:
            // Processing along the horizontal axis (Y)
            switch (coord[1]) {
                case 0:
                case 1:
                    exchange_data(coord[0], coord[1] + 1, &other_data, &other_rank, data, best_rank, !is_sending, comm);
                    if (coord[1] > 0) {
                        exchange_data(coord[0], coord[1] - 1, &other_data, &other_rank, data, best_rank, is_sending, comm);
                    }
                    break;

                case 3:
                case 4:
                    exchange_data(coord[0], coord[1] - 1, &other_data, &other_rank, data, best_rank, !is_sending, comm);
                    if (coord[1] < 4) {
                        exchange_data(coord[0], coord[1] + 1, &other_data, &other_rank, data, best_rank, is_sending, comm);
                    }
                    break;

                default:
                    // Sending along both axes
                    exchange_data(coord[0], coord[1] - 1, &other_data, &other_rank, data, best_rank, is_sending, comm);
                    exchange_data(coord[0], coord[1] + 1, &other_data, &other_rank, data, best_rank, is_sending, comm);
                    break;
            }
            // Sending vertically
            exchange_data(coord[0] - 1, coord[1], &other_data, &other_rank, data, best_rank, is_sending, comm);
            exchange_data(coord[0] + 1, coord[1], &other_data, &other_rank, data, best_rank, is_sending, comm);
            break;
    }
}
