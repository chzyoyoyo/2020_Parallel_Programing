#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>


int su_rand(unsigned int seed)
{
    int32_t val = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
    return val;
}

long long int monte(long long int th_tosses, int world_rank)
{

    // srand( time(NULL) );
    unsigned int seed = time(NULL);

    // seed *= world_rank;

    int th_seed = seed*world_rank;
    // th_seed = su_rand(th_seed)+1;

    long long int th_in_circle = 0;
    long long int num_tosses = th_tosses;

    int sign;
    double x, y;
    double distance_squared;

    // cout << num_tosses << endl;

    // cout << seed <<endl;
    for (int i = 0; i < num_tosses; ++i)
    {

        x = (double) su_rand(th_seed) / RAND_MAX;
        // sem_wait(&sem_seed);
        th_seed = su_rand(th_seed);

        y = (double) su_rand(th_seed) / RAND_MAX;

        th_seed = su_rand(th_seed);

        
        distance_squared = x * x + y * y;
        if ( distance_squared <= 1)
            th_in_circle++;
    }
    return th_in_circle;
}


int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---
    long long int number_in_circle = 0;

    // TODO: init MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    long long int th_tosses = tosses/world_size;
    if (world_rank > 0)
    {
        // TODO: handle workers
        // printf("%d\n", th_tosses);
        number_in_circle = monte(th_tosses, world_rank);
        MPI_Send(&number_in_circle, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);

    }
    else if (world_rank == 0)
    {
        // TODO: master
        th_tosses = tosses - (th_tosses*(world_size-1));
        // printf("%d\n", th_tosses);
        number_in_circle = monte(th_tosses, world_rank);
        for (int i = 1; i < world_size; ++i)
        {
            long long int number;
            MPI_Recv(&number, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
            // printf("number:%d\n", number);
            // printf("number:---------------------");
            number_in_circle += number;
        }
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        pi_result = 4 * number_in_circle /(( double )tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
