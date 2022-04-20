#include <cstdlib>
#include <iostream>
// #include <random>
#include <ctime>

using namespace std;

int main()
{
	srand( time(NULL) );

	int sign;
	double x, y;
	double distance_squared;
	long long int number_in_circle = 0;
	// long long int number_of_tosses = 600000000;//3.1415
	long long int number_of_tosses = 10000000;//3.14xx
	double pi_estimate = 3.1415926;

	while(pi_estimate >= 3.14 && pi_estimate < 3.15)
	{
		for (int i = 0; i < number_of_tosses; ++i)
		{
			/* 產生 [-1 , 1) 的浮點數亂數 */
			sign = rand() % 2;
			x = (double) rand() / RAND_MAX;
			if (sign == 0)
			{
				x = -x;
			}

			sign = rand() % 2;
			y = (double) rand() / RAND_MAX;
			if (sign == 0)
			{
				y = -y;
			}
			// printf("x = %f\n", x);
			// printf("sign = %d\n", sign);
			// printf("y = %f\n", y);
			distance_squared = x * x + y * y;
			if ( distance_squared <= 1)
	        	number_in_circle++;
		}
		
		pi_estimate = 4 * number_in_circle /(( double ) number_of_tosses);
		printf("Pi = %f\n", pi_estimate);
		number_in_circle = 0;
	}

	// printf("RAND_MAX = %d\n", RAND_MAX);
}