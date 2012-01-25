#ifndef FILTERBANKS_H
#define FILTERBANKS_H

float hz2bark(float hz);
float hz2mel(float hz);
float mel2hz(float mel);

void barkweights(int nfreqs, int Fs, float dB, float* wts, int* indices, int* nbands);
void melweights(int nfreqs, int Fs, float dB, float* wts, int* indices, int* nbands);
void linweights(int nfreqs, int Fs, float dB, float** wts, int** indices, int* nbands);

#endif/*FILTERBANKS_H*/
