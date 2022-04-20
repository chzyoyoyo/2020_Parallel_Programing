#include <cstdlib>
#include <iostream>
// #include <random>
#include <ctime>
#include <pthread.h>
#include <unistd.h>
#include <semaphore.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

// static int32_t randtbl[31] =
// {
// 	-1726662223, 379960547, 1735697613, 1040273694, 1313901226,
// 	1627687941, -179304937, -2073333483, 1780058412, -1989503057,
// 	-615974602, 344556628, 939512070, -1249116260, 1507946756,
// 	-812545463, 154635395, 1388815473, -1926676823, 525320961,
// 	-1009028674, 968117788, -123449607, 1284210865, 435012392,
// 	-2017506339, -911064859, -370259173, 1132637927, 1398500161,
// 	-205601318,
// };

long long int number_in_circle = 0;
sem_t penable;
sem_t sem_seed;
int cpu_cores;
unsigned int seed = time(NULL);

int su_rand(unsigned int seed)
{
	int32_t val = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
	return val;
}

void* monte(void* th_tosses)
{

	// srand( time(NULL) );

	sem_wait(&sem_seed);
	seed = su_rand(seed)+6;
	sem_post(&sem_seed);

	int th_seed = seed;
	// th_seed = su_rand(th_seed)+1;

	long long int th_in_circle = 0;
	long long int num_tosses = (long long int) th_tosses;

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

	sem_wait(&penable);
	number_in_circle += th_in_circle;
	sem_post(&penable);
	return 0;
}

int main(int argc, char** argv)
{

	cpu_cores = atoi(argv[1]);
	long long int number_of_tosses = atoi(argv[2]);
	long thread;
	struct timeval start;
	struct timeval end;
	long sec;
	long usec;
	double pi_estimate = 0;
	long long int number_of_tosses_2 = number_of_tosses;


	sem_init(&penable, 0, 1);
	sem_init(&sem_seed, 0, 1);
	// cout << "penable: " << penable << endl;
	pthread_t *thread_handles;
	thread_handles = (pthread_t*)malloc(sizeof(pthread_t)*cpu_cores);
	
	for (thread = 0; thread < cpu_cores-1; thread++) 
	{
		int th_tosses;
		// if (thread < cpu_cores-2)
		// {
		th_tosses = number_of_tosses_2/cpu_cores;
		number_of_tosses -= th_tosses;

		pthread_create(&thread_handles[thread], NULL, monte, (void*) th_tosses);

	}

	// cout << number_of_tosses << endl;
	monte((void*) number_of_tosses);
	
	for (thread = 0; thread < cpu_cores-1; thread++) 
	{
		pthread_join(thread_handles[thread], NULL);
	}
	pi_estimate = 4 * number_in_circle /(( double ) number_of_tosses_2);
	printf("%f\n", pi_estimate);

	// gettimeofday(&end, 0);

	sec = end.tv_sec - start.tv_sec;
    usec = end.tv_usec - start.tv_usec;
	return 0;

}