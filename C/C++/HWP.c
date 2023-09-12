#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[])
{
	int nt  = 4 ;
	# pragma omp parallel num_threads(nt)
	{
		int current_thread = omp_get_thread_num();
		printf("Hello world from thread %d of %d \n",current_thread,nt);
	}
	return EXIT_SUCCESS;
}
