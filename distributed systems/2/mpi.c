#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <sys/time.h>
#include <signal.h>
#include <setjmp.h>
#include<unistd.h>

#define  Max(a,b) ((a)>(b)?(a):(b))
#define  N   (2*2*2*2*2*2+2)
#define EXTRA_PROC 2
#define CHECKPOINT_FILE "checkpoint.dat"

double   maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double eps;
double *local_B, *local_A;
int *recvcounts, *dislps;
int it, rank, size;
int start, end;
int rows;
jmp_buf jp;
MPI_Comm communicator;

void relax(int start, int end, int rank);
void resid(int start, int end, int rank);
void init(int start, int end, int rank);
void verify(int start, int end, int rank);
void write_checkpoint(int rank, int size, double *local_A, int rows);
void read_checkpoint(int rank, int size, double *local_A, int rows);
void handler(MPI_Comm *comm, int *errcode, ...);
    


void exchange_rows(int rank, int size, int rows){
    
    MPI_Request request;
    if (rank < size){
        if (rank != 0)
            MPI_Isend(local_A + 2 * N, 2 * N, MPI_DOUBLE, rank - 1, 0, communicator, &request);
        if (rank != size - 1)
            MPI_Isend(local_A + rows * N, 2 * N, MPI_DOUBLE, rank + 1, 0, communicator, &request);
        if (rank != 0)
            MPI_Irecv( local_A , 2*N , MPI_DOUBLE, rank - 1 , 0 , communicator , &request);
        if (rank != size-1)
            MPI_Irecv( local_A + (rows+2)*N , 2*N , MPI_DOUBLE, rank + 1 , 0 , communicator , &request);
    }
    MPI_Barrier(communicator);
}


int main(int argc, char **argv)
{
    communicator = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Errhandler err_handler;
    MPI_Comm_create_errhandler(handler, &err_handler);
    MPI_Comm_set_errhandler(communicator, err_handler);
    MPI_Comm_rank(communicator, &rank); 
    MPI_Comm_size(communicator, &size); 
    
    size -= EXTRA_PROC;
    MPI_Status status;
    struct timeval starttime, stoptime;
    gettimeofday(&starttime, NULL);
    
    rows = N%size > rank ? N / size + 1: N / size;
    local_A = (double*)malloc(N*(rows+4)*sizeof(double));
    local_B = (double*)malloc(N*rows*sizeof(double));


    start = rank == 0? 2 : rank*(N/size);
    if (N%size > rank){
        start += rank;
    } else {
        start += N % size;
    }

    end = (rank+1)*(N/size);
    if (N%size > rank){
        end += rank;
    } else {
        end += N % size-1;
    }
    
    if (rank == size-1){
        end = N-3;
    }
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
        if (rank < size)
            relax(start, end, rank);
        if (rank == 0)
            resid(start-1,end,rank);
        else{
            if (rank == size-1)
                resid(start,end+1, rank);
            else resid(start, end, rank);
        }

		if (it == 88 && rank == 1){
			int cursize;
			MPI_Comm_size(communicator, &cursize);
			if (cursize == size - EXTRA_PROC)
			raise(SIGKILL);

		}
		if (it == 88 && rank == 1)
			printf("start:%d stop%d\n", start, end);
        
        
        
        if (rank == 0)
			printf("it=%4i   eps=%f\n", it, eps);

        if (eps < maxeps && rank < size)
            break;

        setjmp(jp);
        MPI_Barrier( communicator);
		write_checkpoint(rank, size, local_A, rows);

    }
	if (rank == 0)
	    verify(start-2, end, rank);
    else {
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
    if (rank >= size) return;
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
    if (rank < size){
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
    }
    MPI_Allreduce(&tmp, &eps, 1, MPI_DOUBLE, MPI_MAX, communicator);

}

void verify(int start,int end, int rank)
{
    double s ,tmp = 0.;
    if (rank < size)
		for (i = start; i <= end; i++)
		{
			for (j = 0; j <= N-1; j++)
			{
				tmp += local_A[(i+2-start)*N+j]*(i+1)*(j+1)/(N*N);;
			}
		}
    
    MPI_Reduce(&tmp, &s, 1, MPI_DOUBLE, MPI_SUM, 0, communicator);
    if (rank == 0)
    printf("  S = %f\n",s);
    
}

void write_checkpoint(int rank, int size, double *local_A, int rows) {
    MPI_File fh;
    MPI_Offset offset;
    MPI_Status status;
	
	MPI_Barrier(communicator);

	const char *filename = "checkpoint.dat";
    MPI_File_open(communicator, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	
    offset = rank * (rows + 5) * N * sizeof(double);
	if (it == 87){
	printf("Proc %d saved at offset %lld\n", rank, offset);
	}

	if (rank < size)
		MPI_File_write_at(fh, offset, local_A, (rows+4) * N, MPI_DOUBLE, &status);

    MPI_File_close(&fh);

	MPI_Barrier(communicator);
}

void read_checkpoint(int rank, int size, double *local_A, int rows) {
    MPI_File fh;
    MPI_Status status;
	MPI_Offset offset;

	const char *filename = "checkpoint.dat";
    MPI_File_open(communicator, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    // размер массива каждого процесса
    size_t elements = N * (rows+4);										
    offset = rank * (rows + 5) * N * sizeof(double);

	if (rank < size)
		MPI_File_read_at(fh, offset, local_A, elements, MPI_DOUBLE, &status);

    MPI_File_close(&fh);
    printf("Rank %d: Checkpoint loaded from %s at offset %lld.\n", rank, CHECKPOINT_FILE, offset);
	MPI_Barrier(communicator);
}

void handler(MPI_Comm *comm, int *errcode, ...){
    MPIX_Comm_shrink(communicator, &communicator); // 0 1 2 3 -- упал 1 стало 0 2 3 --> 0 1 2
                                                   //                                   2 3 0
    MPI_Barrier(communicator);
    int cursize;
    MPI_Comm_size(communicator, &cursize);
	MPI_Comm_rank(communicator, &rank);
    if (cursize < size){
        MPI_Abort( communicator, 88);
    }
    // переопределение всего что в начале
    start = rank == 0? 2 : rank*(N/size);
    if (N%size > rank){
        start += rank;
    } else {
        start += N % size;
    }

    end = (rank+1)*(N/size);
    if (N%size > rank){
        end += rank;
    } else {
        end += N % size-1;
    }
    
    if (rank == size-1){
        end = N-3;
    }
    // локальные матрицы прошлой итерации
    read_checkpoint(rank, size, local_A, rows);
	it--;
    // в начало цикла
    longjmp(jp, 0);
}
