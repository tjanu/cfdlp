#define _GNU_SOURCE /* for exp10 */
#include "util.h"
#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>

#include "cfdlp.h"

int same_vectors(int *vec1, int n1, int *vec2, int n2)
{
  if(n1 != n2) return 0;
  for(int i=0;i<n1;i++)
    if(vec1[i] != vec2[i]) return 0;
  return 1;
}

void make_cumhist(int *cumhist, int *hist, int n)
{
  cumhist[0]=0;
  for(int i=0; i<n; i++)
    cumhist[i+1] = cumhist[i] + hist[i];
}

char *mmap_file(char *fn, int *len)
{
  char *result;
  struct stat stat_buf;
  int fd = open(fn, O_RDONLY);

  if(fstat(fd, &stat_buf) == EOF) {
    fprintf(stderr, "mmapfile: can't stat %s\n", fn);
    fatal("mmapfile failed");
  }
  *len = stat_buf.st_size;
  result = mmap(NULL, *len, PROT_READ, MAP_PRIVATE, fd, 0);
  if(result == (char *) EOF) fatal("mmapfile failed");
  close(fd);
  return result;
}

int lenchars_file(char *fn)
{
  FILE *fd = fopen(fn, "r");
  if (fd == NULL) {
    fprintf(stderr, "Unable to open file <%s>: ", fn);
    perror("");
    exit(1);
  }
  if(fseek(fd, 0, SEEK_END) == EOF) fatal("seek failed");
  int len = ftell(fd);
  fclose(fd);
  return len;
}

char *readchars_file(char *fn, long offset, int *nread)
{
   FILE *fd = fopen(fn, "r");
   if (fd == NULL) {
       fprintf(stderr, "Unable to open file <%s>: ", fn);
       perror("");
       exit(1);
   }
   if(fseek(fd, offset, SEEK_SET) == EOF) fatal("seek failed");
   
   char *result = MALLOC(*nread);
   memset(result, 0, *nread);
   int nbytesread = fread(result, 1, *nread, fd);
   //fprintf(stderr,"nread = %d\n",nbytesread);
   
   if(nbytesread != *nread)
      fatal("fread failed");
   
   fclose(fd);
   return result;
}

float *readfeats_file(char *fn, int D, int *fA, int *fB, int *N) 
{
   int nbytes = lenchars_file(fn);
   int nfloats = nbytes/sizeof(float);

   if ( nfloats % D != 0 )
      fatal("ERROR: Feature dimension inconsistent with file\n");

   int nframes = nfloats/D;

   //fprintf(stderr,"nframes = %d\n",nframes);

   if ( *fA == -1 ) *fA = 0;
   if ( *fA >= nframes ) {
      fatal("ERROR: fA is larger than total number of frames\n");
   }

   if ( *fB == -1 ) *fB = nframes-1;
   if ( *fB < *fA ) {
      fatal("ERROR: fB is less than fA\n");
   }
      
   *N = *fB - *fA + 1;
   //fprintf(stderr,"nreq = %d\n",*N);
   int nbytesread = (*N)*D*sizeof(float); 

   return (float*) readchars_file(fn, (*fA)*D*sizeof(float), &nbytesread);
}

short *readsignal_file(char *fn, int *N) 
{
   int nbytes = lenchars_file(fn);
   *N = nbytes/sizeof(short);
   short *data = (short *)readchars_file(fn, 0, &nbytes);
   //float *signal = MALLOC(sizeof(float) * (*N));
   //for (int i = 0; i < *N; i++) {
   // signal[i] = (float)data[i];
   //}
   //FREE(data);
   return data;
}

void writefeats_file(char *fn, float *feats, int D, int nframes) 
{
   FILE *fd = fopen(fn, "w");
   if (fd == NULL) {
       perror("Unable to open output file for writing");
   } else {
       fwrite(feats, D*nframes, sizeof(float), fd);
       fclose(fd);
   }
}

void printfeats_file(char *fn, float *feats, int D, int nframes) 
{
   FILE *fd = fopen(fn, "w");
   for(int i=0; i<nframes; i++){
     for(int j=0; j<D; j++){
       fprintf(fd, "%e ", feats[i*D+j]);
     }
     fprintf(fd, "\n");
   }
   fclose(fd);
}

static int malloc_count = 0;

void *MALLOC(size_t sz)
{
   void *ptr = malloc(sz);

   if(NULL == ptr) fatal("malloc failed\n");
   else malloc_count++;

   return ptr;
}

void FREE(void *ptr)
{
   if (NULL == ptr) fatal("Attempt to free NULL pointer\n");
   else free(ptr);

   malloc_count--;
   return;
}

int get_malloc_count()
{
   return malloc_count;
}

long primep(long base)
{
  long end = sqrt((double)base);
  for (long n = 2; n<end; n++)
    if (base % n == 0) return 0;
  return 1;
}
	
long closest_prime(long base)
{
  for(;;base++)
    if(primep(base))
      return base;
}

int min( int a, int b)
{
   return (a < b) ? a : b;
}

int max(int a, int b)
{
   return (a > b) ? a : b;
}

void fatal(char * msg)
{
   fprintf(stderr, "%s\n", msg);
   exit(2);
}


void threshold_vector(float *vec, int n, double T)
{
  float *end = vec + n;
  for( ; vec < end; vec++)
    *vec = *vec > T;
}

static double lasttime = -1;

void tic( void )
{
   struct timeval time;
   gettimeofday(&time, NULL);
   lasttime = time.tv_sec + ((double)time.tv_usec)/1000000.0;
}

float toc( void )
{
   struct timeval time;
   gettimeofday(&time, NULL);
   double newtime = time.tv_sec + ((double)time.tv_usec)/1000000.0;

   if ( lasttime == -1 ) 
      lasttime = newtime;

   return (float)(newtime - lasttime);
}

int str_to_int(char *str, char* argname)
{
    char *endptr;
    long val = -1;
    int base = 10;

    errno = 0;
    val = strtol(str, &endptr, base);
    if ((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN))
	    || (errno != 0 && val == 0)) {
	fprintf(stderr, "Error when parsing option %s: ", argname);
	perror("strtol");
    }

    if (endptr == str) {
	fprintf(stderr, "No digits were found when parsing option %s\n", argname);
    }

    if (*endptr != '\0')
    {
	fprintf(stderr, "Excess characters after integer argument of option %s: %s\n", argname, endptr);
    }

    return (int)val;
}

float str_to_float(char *str, char *argname)
{
    char *endptr;
    float val = 0.;

    errno = 0;
    val = strtof(str, &endptr);
    if (errno != 0) {
	fprintf(stderr, "Error when parsing option %s: ", argname);
	perror("strtof");
    }

    if (endptr == str) {
	fprintf(stderr, "No digits were found when parsing option %s\n", argname);
    }

    if (*endptr != '\0')
    {
	fprintf(stderr, "Excess characters after float argument of option %s: %s\n", argname, endptr);
    }

    return val;
}

void sdither( short *x, int N, int scale ) 
{
    for ( int i = 0; i < N; i++ )
    {
	float r = ((float) rand())/RAND_MAX;
	x[i] += round(scale*(2*r-1));
    }
}

void fdither_absolute( float *x, int N, int scale ) 
{
    for ( int i = 0; i < N; i++ )
    {
	float r = ((float) rand())/RAND_MAX;
	x[i] += fabs(round(scale*2*r-1));
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

void sub_mean( short *x, int N ) 
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

void pre_emphasize(short *x, int N, float coeff)
{
    if (!(coeff >= 0.0 && coeff < 1.0))
    {
	fprintf(stderr, "WARNING: Not applying pre-emphasis: coefficient (%g) not in range 0 <= coefficient < 1.\n", coeff);
	return;
    }
    if (coeff > 0.0)
    { // actually do something
	for (int i = 1; i < N; i++) { // 1 --> cannot pre-emphasize first sample
	    x[i] = x[i] - coeff * x[i-1];
	}
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

float *fconstruct_frames_wiener( float **x, int *N, int width, int overlap, int *nframes )
{
    *nframes = ceil(((float) *N-width)/(width-overlap)+1);
    int padlen = width + (*nframes-1)*(width-overlap);

    *x = (float *) realloc(*x, padlen*sizeof(float));
    memset(*x+*N, 0, sizeof(float)*(padlen-*N));
    fdither_absolute(*x+*N, padlen-*N, 1);
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

float* log_energies(short **x, int *N, int Fs, int normalize_energies, float silence_floor, float scaling_factor)
{
    int nframes = 0;
    int fwin = DEFAULT_SHORTTERM_WINLEN_MS * Fs;
    int fstep = DEFAULT_SHORTTERM_WINSHIFT_MS * Fs; 
    int add_samp = 0;
    short *frames = sconstruct_frames(x, N, fwin, fwin - fstep, &nframes, &add_samp);

    float* energies = (float*) MALLOC (nframes * sizeof(float));
    float e_max = -1e37;

    // calculate energies and find maximum for normalizing
    for (int f = 0; f < nframes; f++)
    {
	energies[f] = 0.;
	for (int n = 0; n < fwin; n++)
	{
	    energies[f] += frames[f * fwin + n] * frames[f * fwin + n];
	}
	energies[f] = log10f(energies[f]);
	if (energies[f] > e_max)
	{
	    e_max = energies[f];
	}
    }
    float e_floor = e_max / exp10(silence_floor / 10.0f);
    for (int f = 0; f < nframes; f++)
    {
	if (energies[f] < e_floor)
	{
	    energies[f] = e_floor;
	}
	if (normalize_energies == 1)
	{
	    energies[f] = energies[f] - e_max + 1.0f;
	}
	energies[f] *= scaling_factor;
    }
    FREE(frames);
    return energies;
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

