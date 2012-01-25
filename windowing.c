#include "windowing.h"

#include "util.h"
#include <math.h>

// Function to implement generalized hamming window
float * general_hamming(int N, float alpha)
{
    int half; 
    float temp;
    float *x = (float *) MALLOC( N*sizeof(float) );  
    if (N % 2)
	half = (N+1)/2;
    else
	half = N/2;

    for (int i =0; i<N; i++)
    {

	if (i <= half -1)
	{
	    temp = ((float) i)/(N-1);
	    x[i] = alpha - (1 - alpha)*cos(2*M_PI*temp);
	}
	else
	{
	    x[i] = x[N-1-i];
	}
    }

    return x;
}

// Function to implement the hann window
float * hann(int N)
{
    float *x  = (float *) MALLOC(N * sizeof(float) );
    for (int i = 0; i < N+2; i++) {
	if (i > 0 && i < N + 1) { // matlab has no zeroes at the beginning/end for hanning, only for hann (?!)
	    float temp = ((float)i) / (N+1);
	    x[i-1] = 0.5 * (1. - cos(2 * M_PI * temp));
	}
    }
    return x;
}

// Function to implement Hamming Window
float * hamming(int N)
{
    return general_hamming(N, 0.54);
}

