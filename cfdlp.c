#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <complex.h>
#include "util.h"
#include "icsilog.h"
#include <fftw3.h>
#include "adapt.h"

# define pi 3.14159265
void adapt_m(float * in, int N, float fsample, float * out);

/* 0 --> computing fdlpenv the fast way (directly choosing the required number
 * of fft points
 * 1 --> computing fdlpenv the slow way (rounding number of fft points to next
 * power of 2, interpolating for final envelope
 */
#define FDLPENV_WITH_INTERP 0

#define AXIS_BARK 0
#define AXIS_MEL 1
#define AXIS_LINEAR_MEL 2
#define AXIS_LINEAR_BARK 3

char *infile = NULL;
char *outfile = NULL;
char *printfile = NULL;
int Fs = 8000;
int do_gain_norm = 1;
int do_spec = 0;
int axis = 0;
float * dctm = NULL;
float *LOOKUP_TABLE = NULL;
int nbits_log = 14;
int specgrm = 0;
float *fft2decompm = NULL;
float *wts = NULL;
int *indices = NULL;
int nbands = 0;
int auditory_win_length = 0;

int limit_spectrum = 0;

void usage()
{
    fatal("\n USAGE : \n[cfdlp -i <str> -o <str> (REQUIRED)]\n\n OPTIONS  \n -sr <str> Samplerate (8000) \n -gn <flag> -  Gain Normalization (1) \n -spec <flag> - Spectral features (Default 0 --> Modulation features) \n -axis <str> - bark,mel,linear-mel,linear-bark (bark)\n -specgram <flag> - Spectrogram output (0)\n -limit-spectrum <flag> - Limit DCT-spectrum to 125-3800Hz before FDPLP processing\n");
}

void parse_args(int argc, char **argv)
{
    for ( int i = 1; i < argc; i++ )
    {
	if ( strcmp(argv[i], "-i") == 0 )
	{
	    infile = argv[++i];
	}
	else if ( strcmp(argv[i], "-o") == 0 )
	{
	    outfile = argv[++i];
	}
	else if ( strcmp(argv[i], "-print") == 0 )
	{
	    printfile = argv[++i];
	}
	else if ( strcmp(argv[i], "-sr") == 0 )
	{
	    Fs = atoi(argv[++i]);
	}
	else if ( strcmp(argv[i], "-gn") == 0 )
	{
	    do_gain_norm = atoi(argv[++i]);
	}
	else if ( strcmp(argv[i], "-spec") == 0 )
	{
	    do_spec = atoi(argv[++i]);
	}
	else if ( strcmp(argv[i], "-axis") == 0 )
	{
	    i++;
	    if(strcmp(argv[i], "bark") == 0)
	    {
		axis = AXIS_BARK;
	    }
	    else if (strcmp(argv[i], "mel") == 0)
	    {
		axis = AXIS_MEL;
	    }
	    else if (strcmp(argv[i], "linear-mel") == 0)
	    {
		axis = AXIS_LINEAR_MEL;
	    }
	    else if (strcmp(argv[i], "linear-bark") == 0)
	    {
		axis = AXIS_LINEAR_BARK;
	    }
	    else
	    {
		fprintf(stderr, "unknown frequency axis scale: %s\n", argv[i]);
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-specgram") == 0 )
	{
	    specgrm = atoi(argv[++i]); if (specgrm) do_spec = 1;
	}
	else if ( strcmp(argv[i], "-limit-spectrum") == 0 )
	{
	    limit_spectrum = atoi(argv[++i]);
	}
	else
	{
	    fprintf(stderr, "unknown arg: %s\n", argv[i]);
	    usage();
	}
    }

    if ( !infile || !(outfile || printfile) )
    {
	fprintf(stderr, "\nERROR: infile (-i) and at least one of outfile (-o) or printfile (-print) args is required");
	usage();
    }

    if ((axis == AXIS_LINEAR_MEL || axis == AXIS_LINEAR_BARK) && !do_spec)
    {
	fprintf(stderr, "Linear frequency axis is only available for short-term (spectral) features.\n");
	usage();
    }

}

void sdither( short *x, int N, int scale ) 
{
    for ( int i = 0; i < N; i++ )
    {
	float r = ((float) rand())/RAND_MAX;
	x[i] += round(scale*(2*r-1));
    }
}

void fdither( float *x, int N, int scale ) 
{
    for ( int i = 0; i < N; i++ )
    {
	float r = ((float) rand())/RAND_MAX;
	x[i] += round(scale*(2*r-1));
    }
}

void sub_mean( short *x, int N ) 
{
    float sum = 0;
    for ( int i = 0; i < N; i++ )
    {
	sum += x[i];
    }

    short mean = round(sum/N);

    for ( int i = 0; i < N; i++ )
    {
	x[i] -= mean;
    }
}

short *sconstruct_frames( short **x, int *N, int width, int overlap, int *nframes )
{
    *nframes = ceil(((float) *N-width)/(width-overlap)+1);
    int padlen = width + (*nframes-1)*(width-overlap);

    *x = (short *) realloc(*x, padlen*sizeof(short));
    memset(*x+*N, 0, sizeof(short)*(padlen-*N));
    sdither(*x+*N, padlen-*N, 1);
    *N = padlen;

    int step = width-overlap;
    short *frames = MALLOC(*nframes*width*sizeof(short));

    for ( int f = 0; f < *nframes; f++ )
    {
	for ( int n = 0; n < width; n++ )
	{
	    frames[f*width+n] = *(*x + f*step+n);
	}
    }

    return frames;
}

float *fconstruct_frames( float **x, int *N, int width, int overlap, int *nframes )
{
    *nframes = ceil(((float) *N-width)/(width-overlap)+1);
    int padlen = width + (*nframes-1)*(width-overlap);

    *x = (float *) realloc(*x, padlen*sizeof(float));
    memset(*x+*N, 0, sizeof(float)*(padlen-*N));
    fdither(*x+*N, padlen-*N, 1);
    *N = padlen;

    int step = width-overlap;
    float *frames = MALLOC(*nframes*width*sizeof(float));

    for ( int f = 0; f < *nframes; f++ )
    {
	for ( int n = 0; n < width; n++ )
	{
	    frames[f*width+n] = *(*x + f*step+n);
	}
    }

    return frames;
}

// Function to implement Hamming Window
float * hamming(int N)
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
	    x[i] = 0.54 - 0.46*cos(2*pi*temp);
	}
	else
	{
	    x[i] = x[N-1-i];
	}
    }

    return x;
}

// function for implementing deltas
float * deltas(float *x, int nframes, int ncep, int w)
{

    float *d = (float *) MALLOC(ncep*nframes*sizeof(float)); 
    float *xpad = (float *) MALLOC(ncep*(nframes+w-1)*sizeof(float));
    int hlen = floor(w/2);
    for (int fr = 0;fr<(nframes+w-1);fr++)
    {
	for(int cep =0;cep<ncep;cep++)
	{
	    if (fr < hlen)
	    {
		xpad[fr*ncep+cep] = x[cep];
	    }
	    else if (fr >= (nframes+w-1)-hlen)
	    {
		xpad[fr*ncep+cep] = x[(nframes-1)*ncep + cep];
	    }
	    else
	    {
		xpad[fr*ncep+cep] = x[(fr-hlen)*ncep+cep];	
	    }
	}
    }
    for (int fr = w-1;fr<(nframes+w-1);fr++)
    {
	for(int cep =0;cep<ncep;cep++)
	{
	    float temp = 0;	
	    for (int i = 0;i < w; i++)
	    {
		temp += xpad[(fr-i)*ncep+cep]*(hlen-i);	
	    }
	    d[(fr-w+1)*ncep+cep] = temp;
	}
    }
    FREE(xpad);
    return (d);

}

float hz2bark( float hz )
{
    return 6 * asinhf(hz/600);
}

float hz2mel( float hz )
{
    float f_0 = 0; // 133.33333;
    float f_sp = 200/3; // 66.66667;
    float brkfrq = 1000;
    float brkpt  = (brkfrq - f_0)/f_sp;  
    float z;  
    float logstep = exp(log(6.4)/27);

    // fill in parts separately
    if (hz < brkfrq)
    {
	z = (hz - f_0)/f_sp;
    }
    else
    {
	z  = brkpt+((log(hz/brkfrq))/log(logstep));
    }

    return z; 
}

float mel2hz(float mel)
{
  float f_0 = 0;
  float f_sp = 200 / 3;
  float brkfrq = 1000;
  float brkpt = (brkfrq - f_0)/f_sp;
  float z;
  float logstep = exp(log(6.4)/27);

  if (mel < brkpt)
  {
      z = f_0 + f_sp * mel;
  }
  else
  {
      z = brkfrq * exp(log(logstep) * (mel - brkpt));
  }
  return z;
}

void barkweights(int nfreqs, int Fs, float dB, float *wts, int *indices, int *nbands)
{
    // bark per filt
    float nyqbark = hz2bark(Fs/2);
    float step_barks = nyqbark/(*nbands - 1);
    float *binbarks = (float *) MALLOC(nfreqs*sizeof(float));

    // Bark frequency of every bin in FFT
    for ( int i = 0; i < nfreqs; i++ )
    {
	binbarks[i] = hz2bark(((float)i*(Fs/2))/(nfreqs-1));
    }

    for ( int i = 0; i < *nbands; i++ )
    {
	float f_bark_mid = i*step_barks;
	for ( int j = 0; j < nfreqs; j++ )
	{
	    wts[i*nfreqs+j] = exp(-0.5*pow(binbarks[j]-f_bark_mid,2));
	}
    }

    // compute frequency range where each filter exceeds dB threshold
    float lin = pow(10,-dB/20);

    for ( int i = 0; i < *nbands; i++ )
    {
	int j = 0;
	while ( wts[i*nfreqs+(j++)] < lin );
	indices[i*2] = j-1;
	j = nfreqs-1;
	while ( wts[i*nfreqs+(j--)] < lin );
	indices[i*2+1] = j+1;
    }

    FREE(binbarks);
}

void melweights(int nfreqs, int Fs, float dB, float *wts, int *indices, int *nbands)
{
    float nyqmel = hz2mel(Fs/2);
    float step_mels = nyqmel/(*nbands - 1);
    float *binmels = (float *) MALLOC(nfreqs*sizeof(float));

    for ( int i = 0; i < nfreqs; i++ )
    {
	binmels[i] = hz2mel(((float)i*(Fs/2))/(nfreqs-1));
    }

    for ( int i = 0; i < *nbands; i++ )
    {
	float f_mel_mid = i*step_mels;
	for ( int j = 0; j < nfreqs; j++ )
	{
	    wts[i*nfreqs+j] = exp(-0.5*pow(binmels[j]-f_mel_mid,2));
	}
    }

    float lin = pow(10,-dB/20);

    for ( int i = 0; i < *nbands; i++ )
    {
	int j = 0;
	while ( wts[i*nfreqs+(j++)] < lin );
	indices[i*2] = j-1;
	j = nfreqs-1;
	while ( wts[i*nfreqs+(j--)] < lin );
	indices[i*2+1] = j+1;
    }

    FREE(binmels);
}

void linweights(int nfreqs, int Fs, float dB, float **wts, int **indices, int *nbands)
{
    int whop = (int)roundf((float)nfreqs / ((float)(*nbands) + 3.5));
    int wlen = (int)roundf(2.5 * (float)whop);

    int reqbands = ceil(((float)nfreqs-wlen)/(whop + 1));
    *wts = (float *)realloc(*wts, reqbands * nfreqs * sizeof(float));
    *indices = (int *)realloc(*indices, reqbands * 2 * sizeof(int));

    *nbands = reqbands;

    for(int i = 0; i < *nbands; i++)
    {
	for(int j = 0; j < nfreqs; j++)
	{
	    (*wts)[i * nfreqs + j] = 1.;
	}
	(*indices)[i*2] = i * whop;
	(*indices)[i * 2 + 1] = i * whop + wlen;
    }

    if ((*indices)[(*nbands - 1) * 2 + 1] > nfreqs)
    {
	(*indices)[(*nbands - 2) * 2 + 1] = nfreqs;
	(*indices)[(*nbands - 1) * 2] = 0;
	(*indices)[(*nbands - 1) * 2 + 1] = 0;
	*nbands = *nbands - 1;
    }
    else if ((*indices)[(*nbands - 1) * 2 + 1] < nfreqs)
    {
	(*indices)[(*nbands - 1) * 2 + 1] = nfreqs;
    }
}

void levinson(int p, double *phi, float *poles)
{
    double *alpha = (double *) MALLOC((p+1)*(p+1)*sizeof(double));
    double *E = (double *) MALLOC((p+1)*sizeof(double));
    double *k = (double *) MALLOC((p+1)*sizeof(double));
    float g;
    E[0] = phi[0];

    for ( int i = 1; i <= p; i++ )
    {
	k[i] = -phi[i];
	for ( int j = 1; j <= i-1; j++ )
	{
	    k[i] -= (phi[i-j] * alpha[(i-1)*(p+1)+j]);
	}
	k[i] /= E[i-1];

	alpha[i*(p+1)] = 1;
	alpha[i*(p+1)+i] = k[i];
	for ( int j = 1; j <= i-1; j++ )
	{
	    alpha[i*(p+1)+j] = alpha[(i-1)*(p+1)+j] + k[i]*alpha[(i-1)*(p+1)+i-j];
	}
	E[i] = (1-k[i]*k[i])*E[i-1];
    }

    // Copy final iteration coeffs to output array
    g = sqrt(E[p]);

    for ( int i = 0; i <= p; i++ )
    {
	poles[i] = alpha[p*(p+1)+i];
	if (do_gain_norm == 0)
	{
	    //  printf("Gain Normalization Flag is %d ",do_gain_norm);
	    poles[i] /= g; 
	    // printf("Poles %4.4f ",poles[i]);
	}	
	else
	{
	    //	printf("No Gain Normalization ");
	    //	 printf("Poles %4.4f ",poles[i]);
	}
    }
    //printf("Gain Normalization Flag is %d ",do_gain_norm);	
    //printf("Poles %4.4f ",poles[0]);
    //printf("\n");
    FREE(k);
    FREE(E);
    FREE(alpha);
}

void lpc( double *y, int len, int order, int compr, float *poles ) 
{
    // Compute autocorrelation vector or matrix
    int N = pow(2,ceil(log2(2*len-1)));

    double *Y = (double *) MALLOC( N*sizeof(double) );
    for ( int n = 0; n < N; n++ )
    {
	if ( n <= len )
	{
	    Y[n] = y[n];
	}
	else
	{
	    Y[n] = 0;
	}
    }   

    complex *X = (complex *) MALLOC( N*sizeof(complex) );
    fftw_plan plan = fftw_plan_dft_r2c_1d(N, Y, X, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    for ( int n = 0; n < N; n++ )
    {
	X[n] = X[n]*conj(X[n])/len; //add compr
    }

    double *R = (double *) MALLOC( N*sizeof(double) );
    plan = fftw_plan_dft_c2r_1d(N, X, R, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    for ( int n = 0; n < N; n++ )
    {
	R[n] /= N;
    }

    levinson(order, R, poles);

    FREE(R);
    FREE(X);
    FREE(Y);
}

float * fdlpfit_full_sig(short *x, int N, int Fs, int *Np)
{
    double *y = (double *) MALLOC(N*sizeof(double));

    // DEBUG
    //static int framenum = 0;
    //framenum++;

    for ( int n = 0; n < N; n++ )
    {
	y[n] = (double) x[n];
    }

    fftw_plan plan = fftw_plan_r2r_1d(N, y, y, FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    for ( int n = 0; n < N; n++ )
    {
	y[n] /= sqrt(2.0*N);
    }
    y[0] /= sqrt(2);

    int fdlpwin = N;

    float *orig_y = y;
    if (limit_spectrum)
    {
	float lo_freq = 125.;
	float hi_freq = 3800.;

	int lo_offset = round(((float)N/((float)Fs/2.))*lo_freq);
	int hi_offset = round(((float)N/((float)Fs/2))*hi_freq);

	y = y + lo_offset;
	fdlpwin = hi_offset - lo_offset + 1;
    }

    float nyqbar;
    int numbands = 0;
    switch (axis)
    {
	case AXIS_MEL:
	    nyqbar = hz2mel(Fs/2);
	    numbands = ceil(nyqbar)+1;
	    break;
	case AXIS_BARK:
	    nyqbar = hz2bark(Fs/2);
	    numbands = ceil(nyqbar)+1;
	    break;
	case AXIS_LINEAR_MEL:
	case AXIS_LINEAR_BARK:
	    nyqbar = Fs/2;
	    numbands = MIN(96, (int)round((float)N/100.)); // TODO really N or fdlpwin?
	    break;
    }

    if (numbands != nbands || fdlpwin != auditory_win_length) {
	fprintf(stderr, "(Re)creating auditory filter bank (nbands or fdlpwin changed)\n");
	if (wts != NULL) {
	    FREE(wts);
	    wts = NULL;
	}
	if (indices != NULL) {
	    FREE(indices);
	    indices = NULL;
	}
	nbands = numbands;
	auditory_win_length = fdlpwin;
    }

    if (wts == NULL) {
	// Construct the auditory filterbank

	float dB = 48;
	wts = (float *) MALLOC(nbands*auditory_win_length*sizeof(float));
	indices = (int *) MALLOC(nbands*2*sizeof(int));
	switch (axis)
	{
	    case AXIS_MEL:
		melweights(auditory_win_length, Fs, dB, wts, indices, &nbands);
		break;
	    case AXIS_BARK:
		barkweights(auditory_win_length, Fs, dB, wts, indices, &nbands);
		break;
	    case AXIS_LINEAR_MEL:
	    case AXIS_LINEAR_BARK:
		linweights(auditory_win_length, Fs, dB, &wts, &indices, &nbands);
		break;
	}

	// DEBUG
	//fd = fopen("auditory_filterbank.txt", "w");
	//for (int i = 0; i < nbands; i++) {
	//    for (int j = 0; j < fdlpwin; j++) {
	//        fprintf(fd, "%g ", wts[i * fdlpwin + j]);
	//    }
	//    fprintf(fd, "\n");
	//}
	//fclose(fd);
    }

    fprintf(stderr, "Number of sub-bands = %d\n", nbands);	
    switch (axis)
    {
	case AXIS_MEL:
	    *Np = round(N/150);
	    break;
	case AXIS_BARK:
	    *Np = round(N/200);
	    break;
	case AXIS_LINEAR_MEL:
	case AXIS_LINEAR_BARK:
	    *Np = round((indices[1] - indices[0]) / 12);
	    break;
    }
    float *p = (float *) MALLOC( (*Np+1)*nbands*sizeof(float) );

    // Time envelope estimation per band and per frame.
    double *y_filt = (double *) MALLOC(fdlpwin*sizeof(double));
    for ( int i = 0; i < nbands; i++ )
    {
	int Nsub = indices[2*i+1]-indices[2*i]+1;
	for ( int n = 0; n < fdlpwin; n++ )
	{
	    if ( n < Nsub )
	    {
		y_filt[n] = y[indices[2*i]+n] * wts[i*fdlpwin+indices[2*i]+n];
	    }
	    else
	    {
		y_filt[n] = 0;
	    }
	}

	// DEBUG
	//char outname[512];
	//sprintf(outname, "filtered_window_frame-%d_band-%d.txt", framenum, i);
	//FILE *fd = fopen(outname, "w");
	//for (int v = 0; v < Nsub; v++) {
	//    fprintf(fd, "%g ", y_filt[v]);
	//}
	//fclose(fd);

	lpc(y_filt,Nsub,*Np,1,p+i*(*Np+1));
    }

    FREE(y_filt);
    FREE(orig_y);

    // DEBUG
    //char outname[512];
    //sprintf(outname, "poles_frame-%d.txt", framenum);
    //FILE *fd = fopen(outname, "w");
    //for (int i = 0; i < nbands; i++) {
    //    for (int j = 0; j < (*Np+1); j++) {
    //        fprintf(fd, "%g ", p[i * (*Np+1) + j]);
    //    }
    //    fprintf(fd, "\n");
    //}
    //fclose(fd);

    return p;
}

float * fdlpenv( float *p, int Np, int N )
{
    float *env = (float *) MALLOC( N*sizeof(float) );

    int nfft = 2 * (MAX(Np, N) - 1); // --> N = nfft / 2 + 1 == half (fft is symmetric)
    double *Y = (double *) MALLOC( nfft*sizeof(double) );
    for ( int n = 0; n < nfft; n++ )
    {
	if ( n <= Np )
	{
	    Y[n] = p[n];
	}
	else
	{
	    Y[n] = 0;
	}
    }   

    complex *X = (complex *) MALLOC( nfft*sizeof(complex) );
    fftw_plan plan = fftw_plan_dft_r2c_1d(nfft, Y, X, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    for ( int n = 0; n < N; n++ )
    {
	X[n] = 1.0/X[n];
	env[n] = 2*X[n]*conj(X[n]);
    }

    FREE(X);
    FREE(Y);

    return env;
}

float * fdlpenv_mod( float *p, int Np, int N )
{
    float *env = (float *) MALLOC( N*sizeof(float) );

    int nfft = pow(2,ceil(log2(N))+1);
    double *Y = (double *) MALLOC( nfft*sizeof(double) );
    for ( int n = 0; n < nfft; n++ )
    {
	if ( n <= Np )
	{
	    Y[n] = p[n];
	}
	else
	{
	    Y[n] = 0;
	}
    }   

    complex *X = (complex *) MALLOC( nfft*sizeof(complex) );
    fftw_plan plan = fftw_plan_dft_r2c_1d(nfft, Y, X, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    double *h = (double *) MALLOC( nfft*sizeof(double) );

    nfft = nfft/2+1;
    for ( int n = 0; n < nfft; n++ )
    {
	X[n] = 1.0/X[n];
	h[n] = 2*X[n]*conj(X[n]);
    }

    for ( int n = 0; n < N; n++ )
    {
	float p = ((float) n)/(N-1);
	int nleft = floor(p*nfft);
	int nright = ceil(p*nfft);

	float dleft = p - floor(p*nfft)/nfft;
	float dright = ceil(p*nfft)/nfft - p;

	if ( nleft == nright )
	{
	    env[n] = h[nleft];
	}
	else
	{
	    env[n] = (h[nleft]*dright + h[nright]*dleft)/(dleft+dright);
	}
    }


    FREE(h);
    FREE(X);
    FREE(Y);

    return env;
}

void spec2cep(float * frames, int fdlpwin, int nframes, int ncep, int nbands, int band, int offset, float *feats, int log_flag) 
{

    if ( dctm == NULL )
    {
	dctm = (float *) MALLOC(fdlpwin*ncep*sizeof(float));

	for ( int i = 0; i < ncep; i++ )
	{
	    for ( int j = 0; j < fdlpwin; j++ )
	    {
		dctm[i*fdlpwin+j] = cos(M_PI*i*(2.0*((float)j)+1)/(2.0*fdlpwin)) * sqrt(2.0/((float)fdlpwin));
		if ( i == 0 ) 
		{
		    dctm[i*fdlpwin+j] /= sqrt(2);
		}
	    }
	}
    }

    for ( int f = 0; f < nframes; f++ )
    {
	float *frame = frames + f*fdlpwin;
	float *feat =  feats + f*2*ncep*nbands + band*2*ncep + offset;
	for ( int i = 0; i < ncep; i++ )
	{
	    feat[i] = 0;
	    for ( int j = 0; j < fdlpwin; j++ )
	    {
		if ( log_flag )
		{
		    feat[i] += frame[j]*dctm[i*fdlpwin+j];
		}
		else
		{
		    feat[i] += icsi_log(frame[j],LOOKUP_TABLE,nbits_log)*dctm[i*fdlpwin+j];
		}
	    }
	}
    }
}

void spec2cep4energy(float * frames, int fdlpwin, int nframes, int ncep, float *final_feats, int log_flag)
{
    if ( dctm == NULL )
    {
	dctm = (float *) MALLOC(fdlpwin*ncep*sizeof(float));

	for ( int i = 0; i < ncep; i++ )
	{
	    for ( int j = 0; j < fdlpwin; j++ )
	    {
		dctm[i*fdlpwin+j] = cos(M_PI*i*(2.0*((float)j)+1)/(2.0*fdlpwin)) * sqrt(2.0/((float)fdlpwin));
		if ( i == 0 )
		{
		    dctm[i*fdlpwin+j] /= sqrt(2);
		}
	    }
	}
    }
    float *feats = (float *) MALLOC(nframes*ncep*sizeof(float));;
    for ( int f = 0; f < nframes; f++ )
    {
	float *frame = frames + f*fdlpwin;
	float *feat =  feats + f*ncep;
	for ( int i = 0; i < ncep; i++ )
	{
	    feat[i] = 0;
	    for ( int j = 0; j < fdlpwin; j++ )
	    {

		if ( log_flag )
		{
		    feat[i] += frame[j]*dctm[i*fdlpwin+j];
		}
		else
		{
		    feat[i] += (0.33*icsi_log(frame[j],LOOKUP_TABLE,nbits_log))*dctm[i*fdlpwin+j]; //Cubic root compression and log
		    // feat[i] += log(frame[j])*dctm[i*fdlpwin+j];
		}
	    }
	}
    }
    float *del = deltas(feats, nframes,ncep,9);
    float *tempdel = deltas(feats,nframes,ncep,5);
    float *ddel = deltas(tempdel,nframes,ncep,5);
    int dim = 3*ncep;
    for ( int f = 0; f < nframes; f++ )
    {
	for (int cep = 0; cep < dim; cep++)
	{
	    if (cep < ncep)
	    {
		final_feats[f*dim+cep]=feats[f*ncep+cep];
	    }
	    else if (cep < 2*ncep)
	    {
		final_feats[f*dim+cep]=del[f*ncep+cep-ncep];
	    }
	    else
	    {
		final_feats[f*dim+cep]=ddel[f*ncep+cep-2*ncep];
	    }
	}
    }	
    FREE(feats);
    FREE(tempdel);
    FREE(del);
    FREE(ddel);
}

float * fft2melmx(int nfft, int *nbands)
{
    int nfilts = (int)ceilf(hz2mel(Fs/2.)) + 1; 
    *nbands = nfilts;
    float *matrix = (float *) MALLOC(nfilts * nfft * sizeof(float));

    float *fftfreqs = (float *) MALLOC(nfft * sizeof(float));

    float minmel = hz2mel(0);
    float maxmel = hz2mel(Fs/2.);
    float *binfrqs = (float *) MALLOC((nfilts + 2) * sizeof(float));
    int constamp = 0;

    // center freqs of each fft bin
    for (int i = 0; i < nfft; i++)
    {
	fftfreqs[i] = (float)i / (float)nfft * (float)Fs;
    }

    // 'center freqs' of mel bins
    for (int i = 0; i < nfilts + 2; i++)
    {
	binfrqs[i] = mel2hz(minmel + (float)i / ((float)nfilts + 1.) * (maxmel - minmel));
    }

    for (int i = 0; i < nfilts; i++)
    {
	float fs[3];
	fs[0] = binfrqs[i];
	fs[1] = binfrqs[i+1];
	fs[2] = binfrqs[i+2];

	float scale_factor = 2. / binfrqs[i+2] - binfrqs[i];

	for (int j = 0; j < nfft; j++) {
	    if (j <= nfft / 2)
	    {
		float loslope = (fftfreqs[j] - fs[0]) / (fs[1] - fs[0]);
		float hislope = (fs[2] - fftfreqs[j]) / (fs[2] - fs[1]);
		matrix[i * nfft + j] = MAX(0, MIN(hislope, loslope));
		if (constamp == 0)
		{
		    matrix[i * nfft + j] = scale_factor * matrix[i * nfft + j];
		}
	    }
	    else
	    {
		matrix[i * nfft + j] = 0.;
	    }
	}
    }

    FREE(fftfreqs);
    FREE(binfrqs);
    return matrix;
}

float * fft2barkmx(int nfft, int *nbands)
{
    int nfilts = (int)ceilf(hz2bark(Fs/2.)) + 1;
    *nbands = nfilts;
    float *matrix = (float *) MALLOC(nfilts * nfft * sizeof(float));
    float min_bark = hz2bark(0);
    float nyqbark = hz2bark(Fs / 2) - min_bark;
    // bark per filter
    float step_barks = nyqbark / (nfilts-1);
    // frequency of every fft bin in bark
    float *binbarks = (float *) MALLOC(((nfft/2) + 1) * sizeof(float));

    for (int i = 0; i < (nfft/2) + 1; i++)
    {
	binbarks[i] = hz2bark((float)i * (float)Fs / (float)nfft);
    }

    for (int i = 0; i < nfilts; i++)
    {
	float f_bark_mid = min_bark + i * step_barks;
	for (int j = 0; j < nfft; j++)
	{
	    if (j <= nfft / 2)
	    {
		float loslope = (binbarks[j] - f_bark_mid) - 0.5;
		float hislope = (binbarks[j] - f_bark_mid) + 0.5;
		matrix[i * nfft + j] = powf(10.f, MIN(0.f, MIN(hislope, -2.5 * loslope)));
	    }
	    else
	    {
		matrix[i * nfft + j] = 0.;
	    }
	}
    }

    FREE(binbarks);
    return matrix;
}

void audspec(float **bands, int *nbands, int nframes)
{
    float *energybands = *bands;

    int nfft = (*nbands - 1) * 2;
    int nfilts = 0;

    if (fft2decompm == NULL) {
	if (axis == AXIS_LINEAR_MEL) {
	    fft2decompm = fft2melmx(nfft, &nfilts);
	} else if (axis == AXIS_LINEAR_BARK) {
	    fft2decompm = fft2barkmx(nfft, &nfilts);
	} else {
	    fatal("Something went terribly wrong. Trying to convert linear to other band decomposition without having a linear decomp in the first place?.\n");
	}
	
	// DEBUG
	//fprintf(stderr, "Just created the fft2decompm matrix, printing out into file fft2decompm.txt\n");
	//FILE *fft2decompfile = fopen("fft2decompm.txt", "w");
	//for (int i = 0; i < nfilts; i++) {
	//    for (int j = 0; j < nfft; j++) {
	//	fprintf(fft2decompfile, "%g ", fft2decompm[i * nfft + j]);
	//    }
	//    fprintf(fft2decompfile, "\n");
	//}
	//fclose(fft2decompfile);
	//fprintf(stderr, "Done.\n");
    }

//    fprintf(stderr, "Printing out energybands matrix before multiplication into energybands.ascii.\n");
    // DEBUG
    //FILE *ebandsfile = fopen("energybands.txt", "w");
    //for (int i = 0; i < *nbands; i++) {
    //    for (int fr = 0; fr < nframes; fr++) {
    //        fprintf(ebandsfile, "%g ", energybands[fr * (*nbands) + i]);
    //    }
    //    fprintf(ebandsfile, "\n");
    //}
    //fclose(ebandsfile);
//    fprintf(stderr, "Done.\n");

    float *new_bands = (float *) MALLOC (nfilts * nframes * sizeof(float));

    for (int i = 0; i < nfilts; i++)
    {
	for (int f = 0; f < nframes; f++)
	{
	    float temp = 0.;
	    for (int j = 0; j < *nbands; j++)
	    {
		temp += energybands[f * (*nbands) + j] * fft2decompm[i * nfft + j];
	    }
	    new_bands[f * nfilts + i] = temp;
	}
    }

//    fprintf(stderr, "Printing out newbands after multiplication into newbands.ascii\n");
    // DEBUG
    //FILE *nbandsfile = fopen("newbands.txt", "w");
    //for (int i = 0; i < nfilts; i++) {
    //    for (int f = 0; f < nframes; f++) {
    //        fprintf(nbandsfile, "%g ", new_bands[f * nfilts + i]);
    //    }
    //    fprintf(nbandsfile, "\n");
    //}
    //fclose(nbandsfile);
//    fprintf(stderr, "Done.\n");

//    for (int f = 0; f < nframes; f++)
//    {
//	float *frame = energybands + f * *nbands;
//	float *newframe = new_bands + f * nfilts;
//	for (int i = 0; i < nfilts; i++)
//	{
//	    newframe[i] = 0.;
//	    float *transform = fft2decompm + i * nfft;
//	    for (int j = 0; j < *nbands; j++)
//	    {
//		newframe[i] += frame[j] * transform[j];
//	    }
//	}
//    }

    FREE(energybands);
    *bands = new_bands;
    *nbands = nfilts;
}

void compute_fdlp_feats( short *x, int N, int Fs, int nceps, float **feats, int nfeatfr, int numframes, int *dim)
{
    int flen=0.025*Fs;   // frame length corresponding to 25ms
    int fhop=0.010*Fs;   // frame overlap corresponding to 10ms
    int fdlplen = N;
    int fnum = floor((N-flen)/fhop)+1;

    // What's the last sample that feacalc will consider?
    int send = (fnum-1)*fhop + flen;
    int trap = 10;  // 10 FRAME context duration
    int mirr_len = trap*fhop;
    int fdlpwin = 0.2*Fs+flen;  // Modulation spectrum Computation Window.
    int fdlpolap = fdlpwin - fhop;

    int Np;
    float *p = fdlpfit_full_sig(x,fdlplen,Fs,&Np);

    int Npad1 = send+2*mirr_len;
    float *env_pad1 = (float *) MALLOC(Npad1*sizeof(float));

    int Npad2 = send+1000;
    float *env_pad2 = (float *) MALLOC(Npad2*sizeof(float));
    float *env_log = (float *) MALLOC(Npad2*sizeof(float));	
    float *env_adpt = (float *) MALLOC(Npad2*sizeof(float));

    int nframes;
    float *hamm = hamming(flen);  // Defining the Hamming window
    float *energybands = (float *) MALLOC(fnum*nbands*sizeof(float)) ; //Energy of features	

    if (*feats == NULL)
    { // Here we know nbands, finally, and since feats is NULL we are on the first frame

	*dim = nbands*nceps*2;
	if (do_spec)
	{
	    if (specgrm) 
	    {
		*dim = nbands;
		do_spec = 1;
		nceps=nbands;
	    }
	    else
	    {
		nceps=13; 
		*dim = nceps*3;
	    } 
	}

	*feats = (float *)MALLOC(nfeatfr * numframes * (*dim) * sizeof(float));
	fprintf(stderr, "Parameters: (nframes=%d,  dim=%d)\n", numframes, *dim); 
    }

    // DEBUG
    //static int framenum = 0;
    //framenum++;

    for (int i = 0; i < nbands; i++ )
    {
#if FDLPENV_WITH_INTERP == 1
	float *env = fdlpenv_mod(p+i*(Np+1), Np, fdlplen);
#else
	float *env = fdlpenv(p+i*(Np+1), Np, fdlplen);
#endif

	// DEBUG
	//char outname[512];
	//sprintf(outname, "envelope_frame-%d_band-%d.txt", framenum, i);
	//FILE *fd = fopen(outname, "w");
	//for (int v = 0; v < fdlplen; v++) {
	//    fprintf(fd, "%g ", env[v]);
	//}
	//fclose(fd);

	if (do_spec)
	{
	    float *frames = fconstruct_frames(&env, &send, flen, flen-fhop, &nframes);
	    for (int fr = 0; fr < fnum;fr++)
	    {
		float *envwind = frames+fr*flen;
		float temp = 0;
		for (int ind =0;ind<flen;ind++) 
		{
		    temp +=  envwind[ind]*hamm[ind];
		}

		energybands[fr*nbands+i]= temp;

		if (specgrm)
		{
		    (*feats)[fr*nbands+i] = 0.33*log(temp);
		}
	    }
	    FREE(frames);
	    FREE(env);

	}
	else
	{ 
	    for (int k =0;k<N;k++)
	    {
		//env_log[k] = log(env[k]);
		env_log[k] = icsi_log(env[k],LOOKUP_TABLE,nbits_log);     
		sleep(0);	// Found out that icsi log is too fast and gives errors 
	    }

	    //	printf("found_env_log[100] = %4.4f\n",env_log[100]);
	    //	printf("env_log[100] = %4.4f\n",log(env[100]));
	    //	printf("icsi_env_log[100] = %4.4f\n",icsi_log(env[100],LOOKUP_TABLE,nbits_log)) ;

	    for ( int n = 0; n < Npad1; n++ )
	    {
		if ( n < mirr_len )
		{
		    env_pad1[n] = env_log[mirr_len-1-n];
		}
		else if ( n >= mirr_len && n < mirr_len + send )
		{
		    env_pad1[n] = env_log[n-mirr_len];	    
		}
		else
		{
		    env_pad1[n] = env_log[send-(n-mirr_len-send+1)];
		}
	    }

	    float * frames = fconstruct_frames(&env_pad1, &Npad1, fdlpwin, fdlpolap, &nframes);

	    spec2cep(frames, fdlpwin, nframes, nceps, nbands, i, 0, *feats, 1 );

	    FREE(frames);

	    // do delta here
	    float maxenv = 0;
	    for ( int n = 0; n < Npad2; n++ )
	    {
		if ( n < 1000 )
		{
		    env_pad2[n] = env[0];
		}
		else
		{
		    env_pad2[n] = env[n-1000];
		}

		if ( env_pad2[n] > maxenv )
		{
		    maxenv = env_pad2[n];
		}
	    }

	    for ( int n = 0; n < Npad2; n++ )
	    {
		env_pad2[n] /= maxenv;
	    }      

	    adapt_m(env_pad2,Npad2,Fs,env_adpt);
	    //for ( int n = 0; n < Npad2; n++ )
	    //{
	    //} 

	    for ( int n = 0; n < Npad1; n++ )
	    {
		if ( n < mirr_len )
		{
		    env_pad1[n] = env_adpt[mirr_len-1-n+1000];
		}
		else if ( n >= mirr_len && n < mirr_len + send )
		{
		    env_pad1[n] = env_adpt[n-mirr_len+1000];	    
		}
		else
		{
		    env_pad1[n] = env_adpt[send-(n-mirr_len-send+1)+1000];
		}
	    }

	    frames = fconstruct_frames(&env_pad1, &Npad1, fdlpwin, fdlpolap, &nframes);

	    spec2cep(frames, fdlpwin, nframes, nceps, nbands, i, nceps, *feats, 1);  

	    FREE(frames);
	    FREE(env);
	}
    }

    if (do_spec)
    {
	if (axis == AXIS_LINEAR_MEL || axis == AXIS_LINEAR_BARK) {
	    audspec(&energybands, &nbands, nframes);
	}
	if (specgrm)
	{
	    fprintf(stderr,"specgram flag =%d\n",specgrm);
	}
	else
	{
	    spec2cep4energy(energybands, nbands, nframes, nceps, *feats, 0);
	}
    }

    FREE(env_pad1);
    FREE(env_pad2);
    FREE(env_adpt);
    FREE(env_log);
    FREE(p);
    FREE(energybands);
    FREE(hamm);		
}


int main(int argc, char **argv)
{ 
    parse_args(argc, argv);

    LOOKUP_TABLE = (float*) MALLOC(((int) pow(2,nbits_log))*sizeof(float));
    fill_icsi_log_table(nbits_log,LOOKUP_TABLE); 

    int N;
    short *signal = readsignal_file(infile, &N);

    fprintf(stderr, "Input file = %s; N = %d samples\n", infile, N);
    fprintf(stderr, "Gain Norm %d \n",do_gain_norm);
    fprintf(stderr, "Limit Spectrum: %d\n", limit_spectrum);

    int fwin = 0.025*Fs;
    int fstep = 0.010*Fs; 

    int fnum = floor(((float)N-fwin)/fstep)+1;
    N = (fnum-1)*fstep + fwin;

    // DEBUG
    //FILE *fd = fopen("speech_signal.txt", "w");
    //for (int i = 0; i < N; i++) {
    //    fprintf(fd, "%d ", signal[i]);
    //}
    //fclose(fd);

    int fdlpwin = 200*fwin;
    int fdlpolap = 0.020*Fs;  
    int nframes;
    short *frames = sconstruct_frames(&signal, &N, fdlpwin, fdlpolap, &nframes); 

    // DEBUG
    //fd = fopen("speech_frames.txt", "w");
    //for (int i = 0; i < nframes; i++) {
    //    for (int j = 0; j < fdlpwin; j++) {
    //        fprintf(fd, "%d ", frames[i * fdlpwin + j]);
    //    }
    //    fprintf(fd, "\n");
    //}
    //fclose(fd);

    // Compute the feature vector time series
    int nceps = 14;
    int nfeatfr = 498; 
    int dim = 0;

//    float *feats = (float *) MALLOC(nfeatfr*nframes*dim*sizeof(float));
    float *feats = NULL;

    tic();
    for ( int f = 0; f < nframes; f++ )
    {
	short *xwin = frames+f*fdlpwin;
	sdither( xwin, fdlpwin, 1 );
	sub_mean( xwin, fdlpwin );

	compute_fdlp_feats( xwin, fdlpwin, Fs, nceps, &feats + f * nfeatfr * dim, nfeatfr, nframes, &dim );
	printf("\n"); 

	fprintf(stderr, "%f s\n",toc());
    }

    if (outfile)
    {
	fprintf(stderr, "Output file = %s (%d frames, dimension %d)\n", outfile, fnum, dim);
	writefeats_file(outfile, feats, dim, fnum);
    }
    if (printfile)
    {
	fprintf(stderr, "Print file = %s (%d frames, dimension %d)\n", printfile, fnum, dim);
	printfeats_file(printfile, feats, dim, fnum);
    }

    // Free the heap
    FREE(signal);
    FREE(frames);
    FREE(feats);
    FREE(wts);
    FREE(indices);
    if (!(specgrm))
	FREE(dctm);
    if (fft2decompm != NULL)
	FREE(fft2decompm);
    FREE(LOOKUP_TABLE);
    int mc = get_malloc_count();
    if (mc != 0)
	fprintf(stderr,"WARNING: %d malloc'd items not free'd\n", mc);

    return 0;
}
