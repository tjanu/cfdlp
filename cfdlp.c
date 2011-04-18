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

#define AXIS_BARK 0
#define AXIS_MEL 1
#define AXIS_LINEAR 2

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
void usage()
{
    fatal("\n USAGE : \n[cfdlp -i <str> -o <str> (REQUIRED)]\n\n OPTIONS  \n -sr <str> Samplerate (8000) \n -gn <flag> -  Gain Normalization (1) \n -spec <flag> - Spectral features (Default 0 --> Modulation features) \n -axis <str> - bark,mel,linear (bark)\n -specgram <flag> - Spectrogram output (0)\n");
}

void parse_args(int argc, char **argv)
{
    for( int i = 1; i < argc; i++ ) {
	if ( strcmp(argv[i], "-i") == 0 ) infile = argv[++i];
	else if ( strcmp(argv[i], "-o") == 0 ) outfile = argv[++i];
	else if ( strcmp(argv[i], "-print") == 0 ) printfile = argv[++i];
	else if ( strcmp(argv[i], "-sr") == 0 ) Fs = atoi(argv[++i]);
	else if ( strcmp(argv[i], "-gn") == 0 ) do_gain_norm = atoi(argv[++i]);
	else if ( strcmp(argv[i], "-spec") == 0 ) do_spec = atoi(argv[++i]);	
	else if ( strcmp(argv[i], "-axis") == 0 )
	{
	    i++;
	    if(strcmp(argv[i], "bark") == 0)
		axis = AXIS_BARK;
	    else if (strcmp(argv[i], "mel") == 0)
		axis = AXIS_MEL;
	    else if (strcmp(argv[i], "linear") == 0)
		axis = AXIS_LINEAR;
	    else {
		fprintf(stderr, "unknown frequency axis scale: %s\n", argv[i]);
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-specgram") == 0 ) {specgrm = atoi(argv[++i]); if (specgrm) do_spec = 1;}
	else {
	    fprintf(stderr, "unknown arg: %s\n", argv[i]);
	    usage();
	}
    }

    if ( !infile || !outfile || !printfile ) {
	usage();
	fatal("\nERROR: infile (-i), outfile (-o), and printfile (-print) args is required");
    }

}

void sdither( short *x, int N, int scale ) 
{
    for ( int i = 0; i < N; i++ ) {
	float r = ((float) rand())/RAND_MAX;
	x[i] += round(scale*(2*r-1));
    }
}

void fdither( float *x, int N, int scale ) 
{
    for ( int i = 0; i < N; i++ ) {
	float r = ((float) rand())/RAND_MAX;
	x[i] += round(scale*(2*r-1));
    }
}

void sub_mean( short *x, int N ) 
{
    float sum = 0;
    for ( int i = 0; i < N; i++ ) {
	sum += x[i];
    }

    short mean = round(sum/N);

    for ( int i = 0; i < N; i++ ) {
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

    for ( int f = 0; f < *nframes; f++ ) {
	for ( int n = 0; n < width; n++ ) {
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

    for ( int f = 0; f < *nframes; f++ ) {
	for ( int n = 0; n < width; n++ ) {
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
    if (N % 2) half = (N+1)/2;
    else half = N/2;

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
    for (int fr = 0;fr<(nframes+w-1);fr++){
	for(int cep =0;cep<ncep;cep++){
	    if (fr < hlen) xpad[fr*ncep+cep] = x[cep];
	    else if (fr >= (nframes+w-1)-hlen) xpad[fr*ncep+cep] = x[(nframes-1)*ncep + cep];
	    else xpad[fr*ncep+cep] = x[(fr-hlen)*ncep+cep];	
	}}
    for (int fr = w-1;fr<(nframes+w-1);fr++){
	for(int cep =0;cep<ncep;cep++){
	    float temp = 0;	
	    for (int i = 0;i < w;i++) temp += xpad[(fr-i)*ncep+cep]*(hlen-i);	
	    d[(fr-w+1)*ncep+cep] = temp;
	}}
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
    if (hz < brkfrq)   z = (hz - f_0)/f_sp;
    else z  = brkpt+((log(hz/brkfrq))/log(logstep));
    return z; 
}


void barkweights(int nfreqs, int Fs, float dB, float *wts, int *indices, int *nbands)
{
    // bark per filt
    float nyqbark = hz2bark(Fs/2);
    float step_barks = nyqbark/(*nbands - 1);
    float *binbarks = (float *) MALLOC(nfreqs*sizeof(float));

    // Bark frequency of every bin in FFT
    for ( int i = 0; i < nfreqs; i++ ) {
	binbarks[i] = hz2bark(((float)i*(Fs/2))/(nfreqs-1));
    }

    for ( int i = 0; i < *nbands; i++ ) {
	float f_bark_mid = i*step_barks;
	for ( int j = 0; j < nfreqs; j++ ) {
	    wts[i*nfreqs+j] = exp(-0.5*pow(binbarks[j]-f_bark_mid,2));
	}
    }

    // compute frequency range where each filter exceeds dB threshold
    float lin = pow(10,-dB/20);

    for ( int i = 0; i < *nbands; i++ ) {
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

    for ( int i = 0; i < nfreqs; i++ ) {
	binmels[i] = hz2mel(((float)i*(Fs/2))/(nfreqs-1));
    }

    for ( int i = 0; i < *nbands; i++ ) {
	float f_mel_mid = i*step_mels;
	for ( int j = 0; j < nfreqs; j++ ) {
	    wts[i*nfreqs+j] = exp(-0.5*pow(binmels[j]-f_mel_mid,2));
	}
    }

    float lin = pow(10,-dB/20);

    for ( int i = 0; i < *nbands; i++ ) {
	int j = 0;
	while ( wts[i*nfreqs+(j++)] < lin );
	indices[i*2] = j-1;
	j = nfreqs-1;
	while ( wts[i*nfreqs+(j--)] < lin );
	indices[i*2+1] = j+1;
    }

    FREE(binmels);
}

void linweights(int nfreqs, int Fs, float dB, float *wts, int *indices, int *nbands)
{
    int whop = (int)roundf(nfreqs / (*nbands + 3.5));
    int wlen = (int)roundf(2.5 * whop);

    for(int i = 0; i < *nbands; i++) {
	for(int j = 0; j < nfreqs; j++) {
	    wts[i * nfreqs + j] = 1.;
	}
	indices[i*2] = i * whop;
	indices[i * 2 + 1] = i * whop + wlen;
    }

    if (indices[(*nbands - 1) * 2 + 1] > nfreqs) {
	indices[(*nbands - 2) * 2 + 1] = nfreqs;
	indices[(*nbands - 1) * 2] = 0;
	indices[(*nbands - 1) * 2 + 1] = 0;
	*nbands = *nbands - 1;
    } else if (indices[(*nbands - 1) * 2 + 1] < nfreqs) {
	indices[(*nbands - 1) * 2 + 1] = nfreqs;
    }
}

void levinson(int p, double *phi, float *poles)
{
    double *alpha = (double *) MALLOC((p+1)*(p+1)*sizeof(double));
    double *E = (double *) MALLOC((p+1)*sizeof(double));
    double *k = (double *) MALLOC((p+1)*sizeof(double));
    float g;
    E[0] = phi[0];

    for ( int i = 1; i <= p; i++ ) {
	k[i] = -phi[i];
	for ( int j = 1; j <= i-1; j++ ) {
	    k[i] -= (phi[i-j] * alpha[(i-1)*(p+1)+j]);
	}
	k[i] /= E[i-1];

	alpha[i*(p+1)] = 1;
	alpha[i*(p+1)+i] = k[i];
	for ( int j = 1; j <= i-1; j++ ) {
	    alpha[i*(p+1)+j] = alpha[(i-1)*(p+1)+j] + k[i]*alpha[(i-1)*(p+1)+i-j];
	}
	E[i] = (1-k[i]*k[i])*E[i-1];
    }

    // Copy final iteration coeffs to output array
    g = sqrt(E[p]);

    for ( int i = 0; i <= p; i++ ) {
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
    for ( int n = 0; n < N; n++ ) {
	if ( n <= len )
	    Y[n] = y[n];
	else
	    Y[n] = 0;
    }   

    complex *X = (complex *) MALLOC( N*sizeof(complex) );
    fftw_plan plan = fftw_plan_dft_r2c_1d(N, Y, X, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    for ( int n = 0; n < N; n++ ) {
	X[n] = X[n]*conj(X[n])/len; //add compr
    }

    double *R = (double *) MALLOC( N*sizeof(double) );
    plan = fftw_plan_dft_c2r_1d(N, X, R, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    for ( int n = 0; n < N; n++ ) {
	R[n] /= N;
    }

    levinson(order, R, poles);

    FREE(R);
    FREE(X);
    FREE(Y);
}

float * fdlpfit_full_sig(short *x, int N, int Fs, float *wts, int *indices, int nbands, int *Np)
{
    double *y = (double *) MALLOC(N*sizeof(double));

    for ( int n = 0; n < N; n++ ) {
	y[n] = (double) x[n];
    }

    fftw_plan plan = fftw_plan_r2r_1d(N, y, y, FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    for ( int n = 0; n < N; n++ ) {
	y[n] /= sqrt(2.0*N);
    }
    y[0] /= sqrt(2);

    *Np = round(N/150);
    float *p = (float *) MALLOC( (*Np+1)*nbands*sizeof(float) );

    // Time envelope estimation per band and per frame.
    double *y_filt = (double *) MALLOC(N*sizeof(double));	 
    for ( int i = 0; i < nbands; i++ ) {
	int Nsub = indices[2*i+1]-indices[2*i]+1;
	for ( int n = 0; n < N; n++ )
	{
	    if ( n < Nsub )
		y_filt[n] = y[indices[2*i]+n] * wts[i*N+indices[2*i]+n];
	    else
		y_filt[n] = 0;
	}
	lpc(y_filt,Nsub,*Np,1,p+i*(*Np+1));
    }

    FREE(y_filt);
    FREE(y);

    return p;
}

float * fdlpenv_mod( float *p, int Np, int N )
{
    float *env = (float *) MALLOC( N*sizeof(float) );

    int nfft = pow(2,ceil(log2(N))+1);
    double *Y = (double *) MALLOC( nfft*sizeof(double) );
    for ( int n = 0; n < nfft; n++ ) {
	if ( n <= Np ) {
	    Y[n] = p[n];
	} else {
	    Y[n] = 0;
	}
    }   

    complex *X = (complex *) MALLOC( nfft*sizeof(complex) );
    fftw_plan plan = fftw_plan_dft_r2c_1d(nfft, Y, X, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    double *h = (double *) MALLOC( nfft*sizeof(double) );

    nfft = nfft/2+1;
    for ( int n = 0; n < nfft; n++ ) {
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

	if ( nleft == nright ) {
	    env[n] = h[nleft];
	} else {
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

    if ( dctm == NULL ) {
	dctm = (float *) MALLOC(fdlpwin*ncep*sizeof(float));

	for ( int i = 0; i < ncep; i++ ) {
	    for ( int j = 0; j < fdlpwin; j++ ) {
		dctm[i*fdlpwin+j] = cos(M_PI*i*(2.0*((float)j)+1)/(2.0*fdlpwin)) * sqrt(2.0/((float)fdlpwin));
		if ( i == 0 ) 
		    dctm[i*fdlpwin+j] /= sqrt(2);
	    }
	}
    }

    for ( int f = 0; f < nframes; f++ )
    {
	float *frame = frames + f*fdlpwin;
	float *feat =  feats + f*2*ncep*nbands + band*2*ncep + offset;
	for ( int i = 0; i < ncep; i++ ) {
	    feat[i] = 0;
	    for ( int j = 0; j < fdlpwin; j++ ) {

		if ( log_flag )
		    feat[i] += frame[j]*dctm[i*fdlpwin+j];
		else {
		    feat[i] += icsi_log(frame[j],LOOKUP_TABLE,nbits_log)*dctm[i*fdlpwin+j];
		}
	    }
	}
    }
}

void spec2cep4energy(float * frames, int fdlpwin, int nframes, int ncep, float *final_feats, int log_flag)
{
    if ( dctm == NULL ) {
	dctm = (float *) MALLOC(fdlpwin*ncep*sizeof(float));

	for ( int i = 0; i < ncep; i++ ) {
	    for ( int j = 0; j < fdlpwin; j++ ) {
		dctm[i*fdlpwin+j] = cos(M_PI*i*(2.0*((float)j)+1)/(2.0*fdlpwin)) * sqrt(2.0/((float)fdlpwin));
		if ( i == 0 )
		    dctm[i*fdlpwin+j] /= sqrt(2);
	    }
	}
    }
    float *feats = (float *) MALLOC(nframes*ncep*sizeof(float));;
    for ( int f = 0; f < nframes; f++ )
    {
	float *frame = frames + f*fdlpwin;
	float *feat =  feats + f*ncep;
	for ( int i = 0; i < ncep; i++ ) {
	    feat[i] = 0;
	    for ( int j = 0; j < fdlpwin; j++ ) {

		if ( log_flag )
		    feat[i] += frame[j]*dctm[i*fdlpwin+j];
		else {
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
    for ( int f = 0; f < nframes; f++ ){ 
	for (int cep = 0; cep < dim; cep++){
	    if (cep < ncep) final_feats[f*dim+cep]=feats[f*ncep+cep];
	    else if (cep < 2*ncep) final_feats[f*dim+cep]=del[f*ncep+cep-ncep];
	    else final_feats[f*dim+cep]=ddel[f*ncep+cep-2*ncep];
	}}	
    FREE(feats);
    FREE(tempdel);
    FREE(del);
    FREE(ddel);
}


void compute_fdlp_feats( short *x, int N, int Fs, int nbands, int nceps, float *wts, int *indices, float *feats )
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
    float *p = fdlpfit_full_sig(x,fdlplen,Fs,wts,indices,nbands,&Np);

    int Npad1 = send+2*mirr_len;
    float *env_pad1 = (float *) MALLOC(Npad1*sizeof(float));

    int Npad2 = send+1000;
    float *env_pad2 = (float *) MALLOC(Npad2*sizeof(float));
    float *env_log = (float *) MALLOC(Npad2*sizeof(float));	
    float *env_adpt = (float *) MALLOC(Npad2*sizeof(float));

    int nframes;
    float *hamm = hamming(flen);  // Defining the Hamming window
    float *energybands = (float *) MALLOC(fnum*nbands*sizeof(float)) ; //Energy of features	
    for (int i = 0; i < nbands; i++ ) {
	float *env = fdlpenv_mod(p+i*(Np+1), Np, fdlplen);
	if (do_spec){
	    float *frames = fconstruct_frames(&env, &send, flen, flen-fhop, &nframes);
	    for (int fr = 0; fr < fnum;fr++){
		float *envwind = frames+fr*flen;
		float temp = 0;
		for (int ind =0;ind<flen;ind++) 
		    temp +=  envwind[ind]*hamm[ind];

		energybands[fr*nbands+i]= temp;

		if (specgrm) feats[fr*nbands+i] = 0.33*log(temp);

	    }
	    FREE(frames);
	    FREE(env);

	}
	else{ 
	    for (int k =0;k<N;k++)
	    { //env_log[k] = log(env[k]);
		env_log[k] = icsi_log(env[k],LOOKUP_TABLE,nbits_log);     
		sleep(0);	// Found out that icsi log is too fast and gives errors 
	    }

	    //	printf("found_env_log[100] = %4.4f\n",env_log[100]);
	    //	printf("env_log[100] = %4.4f\n",log(env[100]));
	    //	printf("icsi_env_log[100] = %4.4f\n",icsi_log(env[100],LOOKUP_TABLE,nbits_log)) ;

	    for ( int n = 0; n < Npad1; n++ ) {
		if ( n < mirr_len )
		    env_pad1[n] = env_log[mirr_len-1-n];
		else if ( n >= mirr_len && n < mirr_len + send )
		    env_pad1[n] = env_log[n-mirr_len];	    
		else
		    env_pad1[n] = env_log[send-(n-mirr_len-send+1)];
	    }

	    float * frames = fconstruct_frames(&env_pad1, &Npad1, fdlpwin, fdlpolap, &nframes);

	    spec2cep(frames, fdlpwin, nframes, nceps, nbands, i, 0, feats, 1 );

	    FREE(frames);

	    // do delta here
	    float maxenv = 0;
	    for ( int n = 0; n < Npad2; n++ ) {
		if ( n < 1000 ) {
		    env_pad2[n] = env[0];
		} else {
		    env_pad2[n] = env[n-1000];
		}

		if ( env_pad2[n] > maxenv ) {
		    maxenv = env_pad2[n];
		}
	    }

	    for ( int n = 0; n < Npad2; n++ ) {
		env_pad2[n] /= maxenv;
	    }      

	    adapt_m(env_pad2,Npad2,Fs,env_adpt);
	    for ( int n = 0; n < Npad2; n++ ) {
	    } 

	    for ( int n = 0; n < Npad1; n++ ) {
		if ( n < mirr_len )
		    env_pad1[n] = env_adpt[mirr_len-1-n+1000];
		else if ( n >= mirr_len && n < mirr_len + send )
		    env_pad1[n] = env_adpt[n-mirr_len+1000];	    
		else
		    env_pad1[n] = env_adpt[send-(n-mirr_len-send+1)+1000];
	    }

	    frames = fconstruct_frames(&env_pad1, &Npad1, fdlpwin, fdlpolap, &nframes);

	    spec2cep(frames, fdlpwin, nframes, nceps, nbands, i, nceps, feats, 1);  

	    FREE(frames);
	    FREE(env);

	}
    }
    if (do_spec){
	if (specgrm){ fprintf(stderr,"specgram flag =%d\n",specgrm);
	}
	else spec2cep4energy(energybands, nbands, nframes, nceps, feats, 0);
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

    int fwin = 0.025*Fs;
    int fstep = 0.010*Fs; 

    int fnum = floor(((float)N-fwin)/fstep)+1;
    N = (fnum-1)*fstep + fwin;

    int fdlpwin = 200*fwin;
    int fdlpolap = 0.020*Fs;  
    int nframes;
    short *frames = sconstruct_frames(&signal, &N, fdlpwin, fdlpolap, &nframes); 

    // Construct the auditory filterbank
    float nyqbar;
    int nbands = 0;
    switch (axis)
    {
	case AXIS_MEL:
	    nyqbar = hz2mel(Fs/2);
	    nbands = ceil(nyqbar)+1;
	    break;
	case AXIS_BARK:
	    nyqbar = hz2bark(Fs/2);
	    nbands = ceil(nyqbar)+1;
	    break;
	case AXIS_LINEAR:
	    nyqbar = Fs/2;
	    nbands = min(96, (int)roundf(N/100));
	    break;
    }

    float dB = 48;
    float *wts = (float *) MALLOC(nbands*fdlpwin*sizeof(float));
    int *indices = (int *) MALLOC(nbands*2*sizeof(int));
    switch (axis)
    {
	case AXIS_MEL:
	    melweights(fdlpwin, Fs, dB, wts, indices, &nbands);
	    break;
	case AXIS_BARK:
	    barkweights(fdlpwin, Fs, dB, wts, indices, &nbands);
	    break;
	case AXIS_LINEAR:
	    linweights(fdlpwin, Fs, dB, wts, indices, &nbands);
	    break;
    }

    printf("Number of sub-bands = %d\n",nbands);	
    // Compute the feature vector time series
    int nceps = 14;
    int nfeatfr = 498; 
    int dim = nbands*nceps*2;
    if (do_spec){
	if (specgrm) 
	{ dim = nbands; do_spec = 1;nceps=nbands;}
	else {
	    nceps=13; 
	    dim = nceps*3;
	} 
    }
    float *feats = (float *) MALLOC(nfeatfr*nframes*dim*sizeof(float));
    fprintf(stderr, "Parameters: (nframes=%d,  dim=%d)\n", nframes, dim); 
    tic();
    for ( int f = 0; f < nframes; f++ ) {
	short *xwin = frames+f*fdlpwin;
	sdither( xwin, fdlpwin, 1 );
	sub_mean( xwin, fdlpwin );

	compute_fdlp_feats( xwin, fdlpwin, Fs, nbands, nceps, wts, indices, feats + f*nfeatfr*dim );
	printf("\n"); 

	fprintf(stderr, "%f s\n",toc());
    }

    fprintf(stderr, "Output file = %s (%d frames, dimension %d)\n", outfile, fnum, dim);
    writefeats_file(outfile, feats, dim, fnum);
    fprintf(stderr, "Print file = %s (%d frames, dimension %d)\n", printfile, fnum, dim);
    printfeats_file(printfile, feats, dim, fnum);

    // Free the heap
    FREE(signal);
    FREE(frames);
    FREE(feats);
    FREE(wts);
    FREE(indices);
    if (!(specgrm)) FREE(dctm);
    FREE(LOOKUP_TABLE);
    int mc = get_malloc_count();
    if(mc != 0) fprintf(stderr,"WARNING: %d malloc'd items not free'd\n", mc);

    return 0;
}