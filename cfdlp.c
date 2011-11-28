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

// finally get rid of all those magic numbers...
#define DEFAULT_SHORTTERM_WINLEN_MS 0.025
#define DEFAULT_SHORTTERM_WINSHIFT_MS 0.010
#define DEFAULT_SHORTTERM_SHIFT_PERCENTAGE 0.4
#define DEFAULT_FDLPWIN_SHIFT_MS 0.020
#define FDLPWIN_SEC2SHORTTERMMULT_FACTOR 40
#define DEFAULT_LONGTERM_TRAP_FRAMECTX 10
#define DEFAULT_LONGTERM_WINLEN_MS 0.2

#define DEFAULT_SHORTTERM_NCEPS 13
#define DEFAULT_LONGTERM_NCEPS 14

char *infile = NULL;
char *outfile = NULL;
char *printfile = NULL;
int Fs = 8000;
int do_gain_norm = 1;
int do_spec = 0;
int skip_bands = 0;
int axis = AXIS_BARK;
float * dctm = NULL;
float *LOOKUP_TABLE = NULL;
int nbits_log = 14;
int specgrm = 0;
float *fft2decompm = NULL;
float *wts = NULL;
float *orig_wts = NULL;
int *indices = NULL;
int *orig_indices = NULL;
int nbands = 0;
int auditory_win_length = 0;
int fdplp_win_len_sec = 5;

int limit_range = 0;
int do_wiener = 0;
float wiener_alpha = 0.9;
int truncate_last = 0;

char *vadfile = NULL;
char speechchar = '1';
char nspeechchar = '0';
int have_vadfile = 0;
int vad_grace = 2;
int *vad_labels = NULL;
int num_vad_labels = 0;
int vad_label_start = 0;

int shortterm_do_delta = 1;
int shortterm_do_ddelta = 1;
int longterm_do_static = 1;
int longterm_do_dynamic = 1;
int num_cepstral_coeffs = -1;

void usage()
{
    fatal("\nFDLP Feature Extraction software\n"
	    "USAGE:\n"
	    "cfdlp [options] -i <str> [-o <str> | -print <str>]\n"

	    "\nOPTIONS\n\n"
	    " -h, --help\t\tPrint this help and exit\n"
	    "\nIO options:\n\n"
	    " -i <str>\t\tInput file name. Only signed 16-bit little endian raw files are supported. REQUIRED\n"
	    " -o <str>\t\tOutput file name for raw binary float output. Either this or -print is REQUIRED\n"
	    " -print <str>\t\tOutput file name for ascii output, one frame per line. Either this or -o is REQUIRED\n"
	    " -sr <str>\t\tInput samplerate in Hertz. Only 8000 and 16000 Hz are supported. (8000)\n"

	    "\nWindowing options:\n\n"
	    " -fdplpwin <sec>\tLength of FDPLP window in sec (better for reverberant environments when gain normalization is used: 10) (5)\n"
	    " -truncate-last <flag>\ttruncate last frame if number of samples does not fill the entire fdplp window (speeds up computation but also changes numbers) (0)\n"

	    "\nFeature generation options:\n\n"
	    " -gn <flag>\t\tGain Normalization (1) \n"
	    " -limit-range <flag>\tLimit DCT-spectrum to 125-3800Hz before FDPLP processing (0)\n"
	    " -axis <str>\t\tbark,mel,linear-mel,linear-bark (bark)\n"
	    " -skip-bands <int n>\tWhether or not to skip the first n bands when computing the features (useful value for telephone data: 2) (0)\n"
	    " -feat <flag>\t\tFeature type to generate. (0)\n"
	    "\t\t\t\t0: Long-term modulation features\n"
	    "\t\t\t\t1: Short-term spectral features\n"
	    " -spec <flag>\t\tAlternative legacy name for -feat\n"
	    " -specgram <flag>\tSpectrogram output. If this option is given, -feat will have no effect and processing ends after spectrogram output. (0)\n"
	    " -nceps <int>\t\tNumber of cepstral coefficients to use. (14 for modulation features, 13 for short-term features)\n"
	    " -shortterm-mode <int>\tHow to construct a short-term feature frame. No effect if calculating modulation features. (3)\n"
	    "\t\t\t\t0: Only include the <nceps> cepstral coefficients\n"
	    "\t\t\t\t1: <nceps> cepstral coefficients + <nceps> first-order derivatives\n"
	    "\t\t\t\t2: <nceps> cepstral coefficients + <nceps> second-order derivatives\n"
	    "\t\t\t\t3: <nceps> cepstral coefficients + <nceps> first-order derivatives + <nceps> second-order derivatives\n"
	    " -modulation-mode <int>\tHow to construct a long-term modulation feature frame. No effect if calculating short-term features. (2)\n"
	    "\t\t\t\t0: Only include <nceps> statically compressed modulation coefficients per band\n"
	    "\t\t\t\t1: Only include <nceps> dynamically compressed modulation coefficients per band\n"
	    "\t\t\t\t2: <nceps> statically compressed modulation coefficients + <nceps> dynamically compressed modulation coefficients, per band\n"

	    "\nAdditive noise suppression options:\n\n"
	    " -apply-wiener <flag>\tApply Wiener filter (helps against additive noise) (0)\n"
	    " -wiener-alpha <float>\tsets the parameter alpha of the wiener filter (0.9 for modulation and 0.1 for spectral features)\n"
	    " -vadfile <str>\t\tname of the VAD file to read in. Has to be ascii, one char per frame (not given -> energy-based ad-hoc VAD)\n"
	    " -speechchar <char>\tthe char representing speech in the VAD file ('1')\n"
	    " -nonspeechchar <char>\tthe char representing non-speech in the VAD file ('0')\n"
	    " -vad-grace <int>\tmaximum difference between number of frames in VAD file compared to how many are computed. If there are less frames in the VAD file, the last VAD label gets repeated. (2)\n"
	    );
}

void parse_args(int argc, char **argv)
{
    int wiener_alpha_given = 0;
    for ( int i = 1; i < argc; i++ )
    {
	if ( strcmp(argv[i], "-h") == 0
		|| strcmp(argv[i], "--help") == 0
		|| strcmp(argv[i], "-help") == 0)
	{
	    usage();
	}
	else if ( strcmp(argv[i], "-i") == 0 )
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
	    if (Fs != 8000 && Fs != 16000) {
		fatal("Unsupported sample rate! Only 8000 and 16000 are supported.");
	    }
	}
	else if ( strcmp(argv[i], "-gn") == 0 )
	{
	    do_gain_norm = atoi(argv[++i]);
	}
	else if ( strcmp(argv[i], "-spec") == 0 
		|| strcmp(argv[i], "-feat") == 0
		)
	{
	    do_spec = atoi(argv[++i]);
	    if (do_spec != 0 && do_spec != 1)
	    {
		fprintf(stderr, "Error: -feat: Unsupported feature type!\n");
		usage();
	    }
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
	else if ( strcmp(argv[i], "-limit-range") == 0 )
	{
	    limit_range = atoi(argv[++i]);
	}
	else if ( strcmp(argv[i], "-apply-wiener") == 0 )
	{
	    do_wiener = atoi(argv[++i]);
	}
	else if ( strcmp(argv[i], "-wiener-alpha") == 0 )
	{
	    wiener_alpha = (float)atof(argv[++i]);
	    wiener_alpha_given = 1;
	}
	else if ( strcmp(argv[i], "-fdplpwin") == 0)
	{
	    fdplp_win_len_sec = atoi(argv[++i]);
	}
	else if ( strcmp(argv[i], "-truncate-last") == 0)
	{
	    truncate_last = atoi(argv[++i]);
	}
	else if ( strcmp(argv[i], "-skip-bands") == 0)
	{
	    skip_bands = atoi(argv[++i]);
	}
	else if ( strcmp(argv[i], "-vadfile") == 0)
	{
	    vadfile = argv[++i];
	}
	else if ( strcmp(argv[i], "-speechchar") == 0)
	{
	    speechchar = argv[++i][0];
	}
	else if ( strcmp(argv[i], "-nonspeechchar") == 0)
	{
	    nspeechchar = argv[++i][0];
	}
	else if ( strcmp(argv[i], "-vad-grace") == 0)
	{
	    vad_grace = atoi(argv[++i]);
	}
	else if ( strcmp(argv[i], "-nceps") == 0)
	{
	    num_cepstral_coeffs = atoi(argv[++i]);
	}
	else if ( strcmp(argv[i], "-shortterm-mode") == 0)
	{
	    int shortterm_mode = atoi(argv[++i]);
	    switch (shortterm_mode)
	    {
		case 0:
		    shortterm_do_delta = 0;
		    shortterm_do_ddelta = 0;
		    break;
		case 1:
		    shortterm_do_delta = 1;
		    shortterm_do_ddelta = 0;
		    break;
		case 2:
		    shortterm_do_delta = 0;
		    shortterm_do_ddelta = 1;
		    break;
		case 3:
		    shortterm_do_delta = 1;
		    shortterm_do_ddelta = 1;
		    break;
		default:
		    fprintf(stderr, "Unsupported shortterm-mode parameter!\n");
		    usage();
	    }
	}
	else if ( strcmp(argv[i], "-modulation-mode") == 0)
	{
	    int modmode = atoi(argv[++i]);
	    switch (modmode)
	    {
		case 0:
		    longterm_do_static = 1;
		    longterm_do_dynamic = 0;
		    break;
		case 1:
		    longterm_do_static = 0;
		    longterm_do_dynamic = 1;
		    break;
		case 2:
		    longterm_do_static = 1;
		    longterm_do_dynamic = 1;
		    break;
		default:
		    fprintf(stderr, "Error: Unsupported modulation-mode parameter!\n");
		    usage();
	    }
	}
	else
	{
	    fprintf(stderr, "unknown arg: %s\n", argv[i]);
	    usage();
	}
    }

    if ( !infile || !(outfile || printfile) )
    {
	fprintf(stderr, "\nERROR: infile (-i) and at least one of outfile (-o) or printfile (-print) args is required\n");
	usage();
    }

    if ((axis == AXIS_LINEAR_MEL || axis == AXIS_LINEAR_BARK) && !do_spec)
    {
	fprintf(stderr, "Linear frequency axis is only available for short-term (spectral) features.\n");
	usage();
    }

    if (!wiener_alpha_given && do_spec)
    {
	wiener_alpha = 0.1;
    }

    if (skip_bands < 0)
    {
	fprintf(stderr, "Negative number of bands to skip given - how should that be implemented?!\n");
	usage();
    }

    if (num_cepstral_coeffs == -1)
    {
	num_cepstral_coeffs = (do_spec == 0 ? DEFAULT_LONGTERM_NCEPS : DEFAULT_SHORTTERM_NCEPS);
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
	x[i] += round(scale*2*r-1);
    }
}

void sub_mean( float *x, int N ) 
{
    float sum = 0.;
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

short *sconstruct_frames( short **x, int *N, int width, int overlap, int *nframes, int *add_samp)
{
    *nframes = ceil(((float) *N-width)/(width-overlap)+1);
    int padlen = width + (*nframes-1)*(width-overlap);
    *add_samp = padlen - *N;

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
	    x[i] = alpha - (1 - alpha)*cos(2*pi*temp);
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
	    x[i-1] = 0.5 * (1. - cos(2 * pi * temp));
	}
    }
    return x;
}

// Function to implement Hamming Window
float * hamming(int N)
{
    return general_hamming(N, 0.54);
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
	    poles[i] /= g; 
	}	
    }
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
	if ( n < len )
	{
	    Y[n] = y[n];
	}
	else
	{
	    Y[n] = 0;
	}
    }   

    complex *X = (complex *) MALLOC( N*sizeof(complex) );
    memset(X, 0, N * sizeof(complex)); // fix uninitialized value-issue in multiplication below 
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

void hlpc_wiener(double *y, int len, int order, float *poles, int orig_len, int *vadindices, int Nindices)
{
    int wlen = round(DEFAULT_SHORTTERM_WINLEN_MS* Fs);
    float SP = DEFAULT_SHORTTERM_SHIFT_PERCENTAGE;

    int N = 2 * orig_len - 1;

    double *Y = (double *)MALLOC(N * sizeof(double));
    for (int n = 0; n < N; n++)
    {
	if (n < len)
	{
	    Y[n] = y[n];
	}
	else
	{
	    Y[n] = 0.;
	}
    }
    complex *ENV_cmplx = (complex *)MALLOC(N * sizeof(complex));
    memset(ENV_cmplx, 0, N * sizeof(complex));
    fftw_plan plan = fftw_plan_dft_r2c_1d(N, Y, ENV_cmplx, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    float *ENV = (float *) MALLOC(orig_len * sizeof(float));
    for (int i = 0; i < orig_len; i++)
    {
	ENV[i] = (float)cabs(ENV_cmplx[i]) * (float)cabs(ENV_cmplx[i]);
    }

    int envlen = orig_len;
    int envframes = 0;
    int overlap = wlen - (int)round((float)wlen * SP);
    float *fftframes = fconstruct_frames(&ENV, &envlen, wlen, overlap, &envframes);

    float *X = (float *)MALLOC(envframes * wlen * sizeof(float));
    float *Pn = (float *)MALLOC(wlen * sizeof(float));
    for (int i = 0; i < wlen; i++) {
	Pn[i] = 0;
    }
    for (int i = 0; i < Nindices; i++) {
	int frameindex = vadindices[i];
	for (int j = 0; j < wlen; j++) {
	    Pn[j] += fftframes[frameindex * wlen + j];
	}
    }
    for (int i = 0; i < wlen; i++) {
	Pn[i] /= Nindices;
    }

    for (int f = 0; f < envframes; f++)
    {
	for (int i = 0; i < wlen; i++)
	{
	    float noisy_sample = fftframes[f * wlen + i];
	    float gamma = noisy_sample / Pn[i];
	    float zeta = 0.;
	    if (f > 0)
	    {
		zeta = wiener_alpha * X[(f-1) * wlen + i] / Pn[i] + (1 - wiener_alpha) * (gamma - 1);
	    }
	    else
	    {
		zeta = (1-wiener_alpha) * (gamma - 1);
	    }
	    float G = zeta / (1 + zeta); // wiener filter gain
	    X[f * wlen + i] = G * G * noisy_sample; // obtain clean value
	}
    }

    // reconstruct "signal"
    int shiftwidth = (int)(wlen * SP);
    int siglen = (envframes - 1) * shiftwidth + wlen;
    float *ENV_output = (float *)MALLOC(MAX(orig_len, siglen) * sizeof(float));
    memset(ENV_output, 0, MAX(orig_len, siglen) * sizeof(float));
    float *inv_win = (float *)MALLOC(siglen * sizeof(float));
    memset(inv_win, 0, siglen * sizeof(float));
    for (int i = 0; i < envframes; i++)
    {
	int start = i * shiftwidth;
	float *x = X + i * wlen;
	for (int j = 0; j < wlen; j++) {
	    ENV_output[start + j] += x[j];
	    inv_win[start + j] += 1;
	}
    }
    for (int i = 0; i < MAX(orig_len, siglen); i++)
    {
	if (i < siglen)
	{
	    if (inv_win != 0) {
		ENV_output[i] /= inv_win[i];
	    } else { fprintf(stderr, "inv_win[%d] is zero?!\n", i); }
	}
	else
	{
	    ENV_output[i] = ENV[i];
	}
    }

    N = 2 * siglen - 1;
    complex *ENV_cmplx_op = (complex *)MALLOC(N * sizeof(complex));
    for (int i = 0; i < N; i++) {
	int env_output_index = 0;
	if (i < siglen)
	{
	    env_output_index = i;
	}
	else
	{
	    env_output_index = N - i;
	}
	ENV_cmplx_op[i] = ENV_output[env_output_index] / len;
    }

    double *R = (double *)MALLOC(N * sizeof(double));

    plan = fftw_plan_dft_c2r_1d(N, ENV_cmplx_op, R, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    for ( int n = 0; n < N; n++ )
    {
	R[n] /= N;
    }

    levinson(order, R, poles);

    FREE(Y);
    FREE(ENV_cmplx);
    FREE(ENV);
    FREE(fftframes);
    FREE(Pn);
    FREE(X);
    FREE(ENV_output);
    FREE(inv_win);
    FREE(ENV_cmplx_op);
    FREE(R);
}

int *read_VAD(int N, int Fs, int* Nindices)
{
    int flen = DEFAULT_SHORTTERM_WINLEN_MS * Fs;
    int fhop = DEFAULT_SHORTTERM_WINSHIFT_MS * Fs;
    int fnum = floor((N-flen)/fhop)+1;

    int *indices = MALLOC(sizeof(int) * fnum);
    *Nindices = 0;

    if (vad_label_start + fnum > num_vad_labels) {
	fatal("Not enough VAD labels left?!");
    }
    fprintf(stderr, "read_vad: Returning labels for %d frames starting at %d\n", fnum, vad_label_start);
    for (int i = 0; i < fnum; i++) {
	if (vad_labels[vad_label_start + i] == 0) {
	    indices[(*Nindices)++] = i;
	}
    }
    return indices;
}

int *check_VAD(float *x, int N, int Fs, int *Nindices)
{
    int NB_FRAME_THRESHOLD_LTE = 10;
    float LAMBDA_LTE = 0.97;
    int SNR_THRESHOLD_UPD_LTE = 20;
    int ENERGY_FLOOR = 80;
    int MIN_FRAME = 10;
    float lambdaLTEhigherE = 0.99;
    int SNR_THRESHOLD_VAD = 15;
    int MIN_SPEECH_FRAME_HANGOVER = 4;
    int HANGOVER = 15;

    // initialization
    int nbSpeechFrame = 0;
    float meanEN = 0.;
    int hangOver = 0;

    // frame signal inte 25ms frames with 10ms shift
    int flen = DEFAULT_SHORTTERM_WINLEN_MS * Fs;
    float *w = hann(flen);
    float SP = DEFAULT_SHORTTERM_SHIFT_PERCENTAGE;

    int Ncopy = (Fs == 16000 ? N / 2 : N);
    float *copy = MALLOC(sizeof(float) * Ncopy);
    if (Fs == 8000)
    {
	memcpy(copy, x, sizeof(float) * N);
    }
    else
    {
	// resample...
	for (int i = 0; i < Ncopy; i++) {
	    copy[i] = x[i*2];
	}
    }
    int Nframes = 0;
    int overlap = flen - (int)round((float)flen * SP);
    float *x_fr = fconstruct_frames(&copy, &Ncopy, flen, overlap, &Nframes);

    for (int f = 0; f < Nframes; f++)
    {
	for (int n = 0; n < flen; n++)
	{
	    x_fr[f * flen + n] *= w[n];
	}
    }

    int *indices = MALLOC(sizeof(int) * Nframes);
    *Nindices = 0;

    for (int t = 0; t < Nframes; t++)
    {
	float *x_cur = x_fr + t * flen;
	float lambdaLTE = LAMBDA_LTE;
	if (t < NB_FRAME_THRESHOLD_LTE - 1)
	{
	    lambdaLTE = 1. - (1. / (float)(t+1));
	}
	double sum = 0.;
	for (int i = flen - 80; i < flen; i++)
	{
	    sum += x_cur[i] * x_cur[i];
	}
	double frameEN = 0.5 + 16 / (log(2.)) * (log((64. + sum) / 64.));

	if ((frameEN - meanEN) < SNR_THRESHOLD_UPD_LTE || t < MIN_FRAME - 1)
	{
	    if (frameEN < meanEN || t < MIN_FRAME - 1)
	    {
		meanEN = meanEN + (1 - lambdaLTE) * (frameEN - meanEN);
	    }
	    else
	    {
		meanEN = meanEN + (1 - lambdaLTEhigherE) * (frameEN - meanEN);
	    }
	    if (meanEN < ENERGY_FLOOR)
	    {
		meanEN = ENERGY_FLOOR;
	    }
	}
	if (t > 3)
	{
	    if (frameEN - meanEN > SNR_THRESHOLD_VAD)
	    {
		nbSpeechFrame++;
	    }
	    else
	    {
		if (nbSpeechFrame > MIN_SPEECH_FRAME_HANGOVER)
		{
		    hangOver = HANGOVER;
		}
		nbSpeechFrame = 0;
		if (hangOver != 0)
		{
		    hangOver--;
		}
		else
		{
		    indices[(*Nindices)++] = t;
		}
	    }
	}
	else
	{
	    indices[(*Nindices)++] = t;
	}
    }
    FREE(w);
    FREE(x_fr);
    FREE(copy);
    return indices;
}

float * fdlpfit_full_sig(float *x, int N, int Fs, int *Np)
{
    int NNIS = 0;
    int* NIS = NULL;
    if (do_wiener)
    {
	if (have_vadfile)
	{
	    NIS = read_VAD(N, Fs, &NNIS);
	}
	else
	{
	    NIS = check_VAD(x, N, Fs, &NNIS);
	}
    }
    double *y = (double *) MALLOC(N*sizeof(double));

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

    double *orig_y = y;
    if (limit_range)
    {
	float lo_freq = 125.;
	float hi_freq = 3800.;

	int lo_offset = round(((float)N/((float)Fs/2.))*lo_freq) - 1;
	int hi_offset = round(((float)N/((float)Fs/2))*hi_freq) - 1;

	y = y + lo_offset;
	fdlpwin = hi_offset - lo_offset + 1;
    }

    float nyqbar;
    int numbands = 0;
    static int old_nbands = 0;
    static int bank_nbands = 0;
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
	    numbands = MIN(96, (int)round((float)fdlpwin/100.));
	    if (old_nbands != 0) {
		numbands = old_nbands;
	    }
	    break;
    }

    if (numbands != bank_nbands + skip_bands || fdlpwin != auditory_win_length) {
	fprintf(stderr, "(Re)creating auditory filter bank (nbands or fdlpwin changed)\n");
	if (orig_wts != NULL) {
	    FREE(orig_wts);
	    wts = NULL;
	    orig_wts = NULL;
	}
	if (orig_indices != NULL) {
	    FREE(orig_indices);
	    indices = NULL;
	    orig_indices = NULL;
	}
	bank_nbands = numbands;
	auditory_win_length = fdlpwin;
    }

    if (wts == NULL) {
	// Construct the auditory filterbank

	float dB = 48;
	wts = (float *) MALLOC(bank_nbands*auditory_win_length*sizeof(float));
	indices = (int *) MALLOC(bank_nbands*2*sizeof(int));
	switch (axis)
	{
	    case AXIS_MEL:
		melweights(auditory_win_length, Fs, dB, wts, indices, &bank_nbands);
		break;
	    case AXIS_BARK:
		barkweights(auditory_win_length, Fs, dB, wts, indices, &bank_nbands);
		break;
	    case AXIS_LINEAR_MEL:
	    case AXIS_LINEAR_BARK:
		linweights(auditory_win_length, Fs, dB, &wts, &indices, &bank_nbands);
		break;
	}

	orig_wts = wts;
	orig_indices = indices;
	if (skip_bands && bank_nbands > skip_bands) {
	    wts = &orig_wts[skip_bands * fdlpwin];
	    indices = &orig_indices[skip_bands * 2];
	    bank_nbands -= skip_bands;
	}
	old_nbands = bank_nbands;

    }
    
    nbands = bank_nbands;

    fprintf(stderr, "Number of sub-bands = %d\n", nbands);	
    switch (axis)
    {
	case AXIS_MEL:
	    *Np = round((float)fdlpwin/100.);
	    break;
	case AXIS_BARK:
	    *Np = round((float)fdlpwin/200.);
	    break;
	case AXIS_LINEAR_MEL:
	case AXIS_LINEAR_BARK:
	    *Np = round((float)(indices[1] - indices[0]) / 6.);
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
	if (do_wiener)
	{
	    hlpc_wiener(y_filt, Nsub, *Np, p+i*(*Np+1), N, NIS, NNIS);
	}
	else
	{
	    lpc(y_filt,Nsub,*Np,1,p+i*(*Np+1));
	}
    }

    FREE(y_filt);
    FREE(orig_y);
    if (NIS != NULL)
    {
	FREE(NIS);
    }

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
    memset(h, 0, nfft * sizeof(double));

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

    int dimmult = longterm_do_static + longterm_do_dynamic;
    for ( int f = 0; f < nframes; f++ )
    {
	float *frame = frames + f*fdlpwin;
	float *feat =  feats + f*dimmult*ncep*nbands + band*dimmult*ncep + offset;
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
    memset(feats, 0, nframes * ncep * sizeof(float));
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
		}
	    }
	}
    }
    float *del = NULL;
    if (shortterm_do_delta == 1)
    {
	del = deltas(feats, nframes,ncep,9);
    }
    float *ddel = NULL;
    if (shortterm_do_ddelta == 1)
    {
	float *tempdel = deltas(feats,nframes,ncep,5);
	ddel = deltas(tempdel,nframes,ncep,5);
	FREE(tempdel);
    }
    int dimmult = 1 + shortterm_do_delta + shortterm_do_ddelta;
    int dim = dimmult*ncep;
    for ( int f = 0; f < nframes; f++ )
    {
	for (int cep = 0; cep < dim; cep++)
	{
	    if (cep < ncep)
	    {
		final_feats[f*dim+cep]=feats[f*ncep+cep];
	    }
	    else if (cep < 2*ncep && shortterm_do_delta == 1)
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
    if (del != NULL)
    {
	FREE(del);
    }
    if (ddel != NULL)
    {
	FREE(ddel);
    }
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
    }

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

    FREE(energybands);
    *bands = new_bands;
    *nbands = nfilts;
}

void compute_fdlp_feats( float *x, int N, int Fs, int* nceps, float **feats, int nfeatfr, int numframes, int *dim)
{
    int flen= DEFAULT_SHORTTERM_WINLEN_MS * Fs;   // frame length corresponding to 25ms
    int fhop= DEFAULT_SHORTTERM_WINSHIFT_MS * Fs;   // frame overlap corresponding to 10ms
    int fdlplen = N;
    int fnum = floor((N-flen)/fhop)+1;
    //int fnum = floor((N-flen)/fhop); // bug in feacalc will result in 1 frame less for 8kHzs data than for 16kHz :-/

    // What's the last sample that feacalc will consider?
    int send = (fnum-1)*fhop + flen;
    int trap = DEFAULT_LONGTERM_TRAP_FRAMECTX;  // 10 FRAME context duration
    int mirr_len = trap*fhop;
    int fdlpwin = DEFAULT_LONGTERM_WINLEN_MS * Fs+flen;  // Modulation spectrum Computation Window.
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
	int dimmult = longterm_do_static + longterm_do_dynamic;
	*dim = nbands*(*nceps) * dimmult;
	if (do_spec)
	{
	    if (specgrm) 
	    {
		*dim = nbands;
		do_spec = 1;
		*nceps=nbands;
	    }
	    else
	    {
		//*nceps=DEFAULT_SHORTTERM_NCEPS; // now in argument parsing and
		//main...
		int dimmult = 1 + shortterm_do_delta + shortterm_do_ddelta;
		*dim = (*nceps) * dimmult;
	    } 
	}

	*feats = (float *)MALLOC(nfeatfr * numframes * (*dim) * sizeof(float));
	fprintf(stderr, "Parameters: (nframes=%d,  dim=%d)\n", numframes, *dim); 
    }

    for (int i = 0; i < nbands; i++ )
    {
#if FDLPENV_WITH_INTERP == 1
	float *env = fdlpenv_mod(p+i*(Np+1), Np, fdlplen);
#else
	float *env = fdlpenv(p+i*(Np+1), Np, fdlplen);
#endif

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
		env_log[k] = icsi_log(env[k],LOOKUP_TABLE,nbits_log);     
		sleep(0);	// Found out that icsi log is too fast and gives errors 
	    }

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

	    if (longterm_do_static == 1)
	    {
		spec2cep(frames, fdlpwin, nframes, *nceps, nbands, i, 0, *feats, 1 );
	    }

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

	    if (longterm_do_dynamic == 1)
	    {
		spec2cep(frames, fdlpwin, nframes, *nceps, nbands, i, *nceps * longterm_do_static, *feats, 1);
	    }

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
	    spec2cep4energy(energybands, nbands, nframes, *nceps, *feats, 0);
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
    float *signal = readsignal_file(infile, &N);
    int Nsignal = N;

    fprintf(stderr, "Input file = %s; N = %d samples\n", infile, N);
    fprintf(stderr, "Gain Norm %d \n",do_gain_norm);
    fprintf(stderr, "Limit DCT range: %d\n", limit_range);
    fprintf(stderr, "Apply wiener filter: %d (alpha=%g)\n", do_wiener, wiener_alpha);

    int fwin = DEFAULT_SHORTTERM_WINLEN_MS * Fs;
    int fstep = DEFAULT_SHORTTERM_WINSHIFT_MS * Fs; 

    int fnum = floor(((float)N-fwin)/fstep)+1;
    //int fnum = floor(((float)N-fwin)/fstep); // stupid bug in feacalc...
    N = (fnum-1)*fstep + fwin;

    Nsignal = N;

    int fdlpwin = fdplp_win_len_sec * FDLPWIN_SEC2SHORTTERMMULT_FACTOR * fwin;
    int fdlpolap = DEFAULT_FDLPWIN_SHIFT_MS * Fs;  
    int nframes;
    int add_samp;
    float *frames = fconstruct_frames(&signal, &N, fdlpwin, fdlpolap, &nframes);
    add_samp = N - Nsignal;

    // read in VAD if we have to
    if (vadfile != NULL) {
	num_vad_labels = fnum;
	if (!truncate_last)
	{
	    num_vad_labels = floor(((float)N-fwin)/fstep)+1;
	}
	int num_read_labels = lenchars_file(vadfile);
	char *labels = readchars_file(vadfile, 0, &num_read_labels);
	if (labels[num_read_labels - 1] == '\n') {
	    num_read_labels--;
	}
	fprintf(stderr, "Number of frames to label: %d, number of labels in file: %d\n", num_vad_labels, num_read_labels);
	fprintf(stderr, "VAD grace? %d\n", vad_grace);
	fprintf(stderr, "Truncate_last? %d\n", truncate_last);
	vad_labels = (int *)MALLOC(num_vad_labels * sizeof(int));
	for (int i = 0; i < num_vad_labels; i++) {
	    if (i < num_read_labels) {
		vad_labels[i] = (labels[i] == speechchar ? 1 : (labels[i] == nspeechchar ? 0 : 2));
		if (vad_labels[i] == 2) {
		    fatal("VAD file had unspecified character in it!");
		}
	    } else {
		if (i < num_read_labels + vad_grace)
		{
		    vad_labels[i] = vad_labels[num_read_labels - 1];
		}
		else if (!truncate_last)
		{
		    vad_labels[i] = 0;
		}
		else
		{
		    fatal("VAD file contains too few labels.");
		}
	    }
	}
	FREE(labels);
	have_vadfile = 1;
	vad_label_start = 0;
    }

    // Compute the feature vector time series
    int nceps = num_cepstral_coeffs;
    int dim = 0;
    int nfeatfr_calculated = 0;

    float *feats = NULL;

    tic();
    int stop_before = 0;
    for ( int f = 0; !stop_before && f < nframes; f++ )
    {
	int local_size = fdlpwin;
	if (truncate_last && Nsignal - f * (fdlpwin - fdlpolap) < fdlpwin)
	{
	    local_size = Nsignal - f * (fdlpwin - fdlpolap);
	}
	float *xwin = frames+f*fdlpwin;
	if (f < nframes - 1 && Nsignal - (f + 1) * (fdlpwin - fdlpolap) < 0.2 * Fs)
	{
	    // have at least .2 seconds in the last frame or just enlarge the second-to-last frame
	    local_size = Nsignal + fdlpolap - f * (fdlpwin - fdlpolap);
	    if (local_size > Nsignal)
	    {
		local_size = Nsignal;
	    }
	    stop_before = 1;
	    xwin = signal + (Nsignal - local_size);
	}
	fdither( xwin, local_size, 1 );
	sub_mean( xwin, local_size );

	int nfeatfr = (int)floor((local_size - fwin)/fstep)+1;

	if (feats == NULL)
	{
	    compute_fdlp_feats( xwin, local_size, Fs, &nceps, &feats, nfeatfr, nframes, &dim );
	}
	else
	{
	    float *feat_mem = feats + nfeatfr_calculated * dim;
	    compute_fdlp_feats( xwin, local_size, Fs, &nceps, &feat_mem, nfeatfr, nframes, &dim );
	}
	printf("\n"); 
	nfeatfr_calculated += nfeatfr;
	vad_label_start = nfeatfr_calculated;
	fprintf(stderr, "Just computed %d frames, now have %d frames calculated in total.\n", nfeatfr, nfeatfr_calculated);

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
    if (vad_labels != NULL) {
	FREE(vad_labels);
    }
    FREE(signal);
    FREE(frames);
    FREE(feats);
    FREE(orig_wts);
    FREE(orig_indices);
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
