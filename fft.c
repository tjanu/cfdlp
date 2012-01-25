#include "fft.h"

#include <fftw3.h>
#include <complex.h>

struct fft_info_ {
    fftw_plan plan;
    int plan_size;
    union input {
	double* dbl;
	complex* cmplx;
    };
    union output {
	double* dbl;
	complex *cmplx;
    };
};
