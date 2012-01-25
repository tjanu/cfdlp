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

