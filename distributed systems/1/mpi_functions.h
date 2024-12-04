#ifndef MPI_FUNCTIONS_H
#define MPI_FUNCTIONS_H

#include <mpi.h>

void exchange_data(int other_row, int other_col, 
                 int *recv_data, int *recv_rank, 
                 int *data, int *best_rank, 
                 int send, MPI_Comm comm);

void find_max(int *data, int *best_rank, MPI_Comm comm, int coord[2]);
void send_to_all(int *data, int *best_rank, MPI_Comm comm, int coord[2]);

#endif