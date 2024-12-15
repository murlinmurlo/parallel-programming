## Компиляция и запуск:
Ваша версия MPI должна быть собрана с поддержкой ULFM.

```bash 
mpicc mpi.c -o mpi.o 
mpirun -np 4 mpi.o --enable-recovery --with-ft ulfm  --oversubscribe
