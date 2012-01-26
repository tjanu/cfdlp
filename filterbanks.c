#include "filterbanks.h"
#include "util.h"

#include <math.h>

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

