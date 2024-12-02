#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <sys/time.h>

#define  Max(a,b) ((a)>(b)?(a):(b))
#define  N   (2*2*2*2*2*2+2)
double   maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double eps;
double *local_B, *local_A;
int *recvcounts, *dislps;

void relax(int start, int end, int rank);
void resid(int start, int end, int rank);
void init(int start, int end, int rank);
void verify(int start, int end, int rank);



void exchange_rows(int rank, int size, int rows){
    MPI_Request request;
    if (rank != 0)
        MPI_Isend(local_A + 2 * N, 2 * N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request);
    if (rank != size - 1)
        MPI_Isend(local_A + rows * N, 2 * N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request);
    if (rank != 0)
        MPI_Irecv( local_A , 2*N , MPI_DOUBLE, rank - 1 , 0 , MPI_COMM_WORLD , &request);
    if (rank != size-1)
        MPI_Irecv( local_A + (rows+2)*N , 2*N , MPI_DOUBLE, rank + 1 , 0 , MPI_COMM_WORLD , &request);
    MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char **argv)
{
    int it, rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Status status;
    struct timeval starttime, stoptime;
    gettimeofday(&starttime, NULL);
    int rows = N%size > rank ? N / size + 1: N / size;
    local_A = (double*)malloc(N*(rows+4)*sizeof(double));
    local_B = (double*)malloc(N*rows*sizeof(double));


    int start = rank == 0? 2 : rank*(N/size);
    if (N%size > rank){
        start += rank;
    } else {
        start += N % size;
    }

    int end = (rank+1)*(N/size);
    if (N%size > rank){
        end += rank;
    } else {
        end += N % size-1;
    }
    
    if (rank == size-1)
        end = N-3;
    if (rank == 0)
        init(start-2, end, rank);
    else{
        if (rank == size-1)
        init(start,end+2,rank);
        else 
        init(start, end,rank);
    }
    for (it = 1; it <= itmax; it++)
    {
        exchange_rows(rank, size, rows);
        eps = 0.;
        relax(start, end, rank);
        if (rank == 0)
        resid(start-1,end,rank);
        else{
            if (rank == size-1)
                resid(start,end+1, rank);
            else resid(start, end, rank);
        }
        
        
        
        
        if (rank == 0)
        printf("it=%4i   eps=%f\n", it, eps);

        if (eps < maxeps)
            break;
    }
    if (rank == 0)
        verify(start-2, end, rank);
    else{
        if (rank == size-1)
        verify(start,end+2,rank);
        else 
        verify(start, end,rank);
    }
    if (rank == 0){
        gettimeofday(&stoptime, NULL);
        long seconds = stoptime.tv_sec - starttime.tv_sec;
        long micsec  = stoptime.tv_usec - starttime.tv_usec;
        FILE * F = fopen("mpi_times.txt", "a");
        fprintf(F, "%f\n", seconds + micsec*1e-6);
        }
    

    free(local_A);
    free(local_B);
    MPI_Finalize();
    return 0;
}

void init(int start, int end, int rank)
{ 
    
    for (i = start; i <= end; i++)
    {
        for (j = 0; j <= N-1; j++)
        {
            if (i == 0 || i == N-1 || j == 0 || j == N-1)
                local_A[(i+2-start)*N + j] = 0.;
            else
                local_A[(i+2-start)*N + j] = (1. + i + j);
        }
    }
}

void relax(int start, int end, int rank)
{
    int k = rank == 0 ? 2: 0;
    for (i = start; i <= end; i++)
    {
        for (j = 2; j <= N-3; j++)
        {
            local_B[(i-start+k)*N + j]=(local_A[(i-2+2-start+k) * N + j]+local_A[(i-1+2-start+k)*N + j]+local_A[(i+2+2-start+k)*N +j]+local_A[(i+1+2-start+k)*N + j]+local_A[(i+2-start+k)*N + j-2]+local_A[(i+2-start+k)*N + j-1]+local_A[(i+2-start+k)*N + j+2]+local_A[(i+2-start+k)*N + j+1])/8.;
        }
    }
}

void resid(int start,int end, int rank)
{ 
    double tmp = 0;
    int k = rank == 0 ? 1:0; 
    for (i = start; i <= end; i++)
    {
        for (j = 1; j <= N-2; j++)
        {
            double e;
            e = fabs(local_A[(i+2-start+k)*N + j] - local_B[(i-start+k)*N + j]);
            tmp = Max(tmp,e);
            local_A[(i+2-start+k)*N+j] = local_B[(i-start+k)*N+j];
        }
    }

    MPI_Allreduce(&tmp, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

}

void verify(int start,int end, int rank)
{
    double s ,tmp = 0.;
    
    for (i = start; i <= end; i++)
    {
        for (j = 0; j <= N-1; j++)
        {
            tmp += local_A[(i+2-start)*N+j]*(i+1)*(j+1)/(N*N);;
        }
    }
    
    MPI_Reduce(&tmp, &s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    printf("  S = %f\n",s);
    
}
