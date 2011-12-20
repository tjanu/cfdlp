#ifndef UTIL_H
#define UTIL_H

#include <stdlib.h>

#define M_PI           3.14159265358979323846  /* pi */

int same_vectors(int *vec1, int n1, int *vec2, int n2);
void make_cumhist(int *cumhist, int *hist, int n);

void *MALLOC(size_t sz);
void FREE(void *ptr);
int get_malloc_count( void );
long closest_prime( long base );
int min(int a, int b);
int max(int a, int b);
void fatal(char *msg);
void threshold_vector(float *vec, int n, double T);

int lenchars_file(char* fn);
char* readchars_file(char* fn, long offset, int* nread);

float *readfeats_file(char *fn, int D, int *fA, int *fB, int *N);
short *readsignal_file(char *fn, int *N);
void writefeats_file(char *fn, float *feats, int D, int nframes);
void printfeats_file(char *fn, float *feats, int D, int nframes);
char *mmap_file(char *fn, int *len);

void tic(void);
float toc(void);

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#endif
