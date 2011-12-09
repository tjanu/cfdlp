#define _GNU_SOURCE
#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <complex.h>
#include <errno.h>
#include <limits.h>
#include "util.h"
#include "icsilog.h"
#include <fftw3.h>
#include "adapt.h"

#if HAVE_LIBPTHREAD == 1
#include <pthread.h>
#include <time.h>
#endif

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
char *specfile = NULL;
int verbose = 0;
int Fs = 8000;
int factor = -1;
int Fs1 = -1;
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
float fdplp_win_len_sec = 5;

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

float* Pn_buf = NULL;
int Pn_buf_valid = 0;

// FFTW plans "outsourced"
fftw_plan dct_plan = NULL;
int dct_plan_size = -1;
double *dct_buffer = NULL;

fftw_plan* lpc_r2c_plans = NULL;
int num_lpc_plans = 0;
int* lpc_r2c_plan_sizes = NULL;
double** lpc_r2c_input_buffers = NULL;
complex** lpc_r2c_output_buffers = NULL;
fftw_plan* lpc_c2r_plans = NULL;
int* lpc_c2r_plan_sizes = NULL; // only needed in hlpc_wiener
complex** lpc_c2r_input_buffers = NULL; // only needed in hlpc_wiener
double** lpc_c2r_output_buffers = NULL;

fftw_plan* fdlpenv_plans = NULL;
int num_fdlpenv_plans = 0;
int* fdlpenv_plan_sizes = NULL;
double** fdlpenv_input_buffers = NULL;
complex** fdlpenv_output_buffers = NULL;

// Multithreading
int max_num_threads = 1;
struct thread_info {
#if HAVE_LIBPTHREAD == 1
    pthread_t thread_id;
#endif
    void* (*threadfunc)(void* arg);
    void* thread_arg;
};

#if HAVE_LIBPTHREAD == 1
struct running_thread {
    pthread_t* thread_id;
    struct running_thread* next;
};
#endif

struct thread_info* band_threads = NULL;
#if HAVE_LIBPTHREAD == 1
pthread_mutex_t fftw_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t tpool_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t tpool_cond = PTHREAD_COND_INITIALIZER;
#endif

struct lpc_info {
    double* y;
    int len;
    int order;
    int compression;
    float* poles;
    int orig_len;
    int* vadindices;
    int Nindices;
    int band;
};

struct fdlpenv_info {
    float* poles;
    int Np;
    int fdlplen;
    int ffdlplen;
    int band;
    int fnum;
    int send;
    int fsend;
    int flen;
    int fflen;
    int fhop;
    int ffhop;
    int mirrlen;
    int fdlpwin;
    int fdlpolap;
    int nceps;
    int* nframes;
    float* energybands; // if do_spec
    float* feats; // if !do_spec
    float* spectrogram;
    float* hamm;
    float* fhamm;
};

void run_threads(struct thread_info* threads, int numthreads)
{
    int i = 0;
    int curr_num_threads = 0;
#if HAVE_LIBPTHREAD == 1
    struct running_thread* running_threads = NULL;
    struct timespec ts;
#endif
    //fprintf(stderr, "Got %d jobs to run with a maximum number of %d threads\n", numthreads, max_num_threads);
    while (i < numthreads)
    {
#if HAVE_LIBPTHREAD == 1
	//fprintf(stderr, "Scheduling job %d, %d/%d threads currently running\n", i, curr_num_threads, max_num_threads);
	if (max_num_threads < 0 || curr_num_threads < max_num_threads)
	{
	    //fprintf(stderr, "starting job\n");
	    pthread_create(&threads[i].thread_id, NULL, threads[i].threadfunc, threads[i].thread_arg);
	    struct running_thread* new_runner = (struct running_thread*) MALLOC(sizeof(struct running_thread));
	    new_runner->thread_id = &threads[i].thread_id;
	    new_runner->next = NULL;
	    struct running_thread* list_end = running_threads;
	    if (list_end == NULL)
	    {
		running_threads = new_runner;
	    }
	    else
	    {
		while (list_end != NULL && list_end->next != NULL)
		{
		    list_end = list_end->next;
		}
		list_end->next = new_runner;
	    }
	    curr_num_threads++;
	    i++;
	}
	// clean up running thread list in case anything is already finished
	struct running_thread* last = NULL;
	//fprintf(stderr, "Cleaning up running jobs...\n");
	pthread_mutex_lock(&tpool_mutex);
	if (max_num_threads > 0 && curr_num_threads >= max_num_threads)
	{
	    clock_gettime(CLOCK_REALTIME, &ts);
	    ts.tv_nsec += 5000000; // half a second
	    if (ts.tv_nsec >= 1000000000)
	    {
		int nsecs = ts.tv_nsec / 1000000000;
		ts.tv_sec += nsecs;
		ts.tv_nsec -= nsecs * 1000000000;
	    }
	    pthread_cond_timedwait(&tpool_cond, &tpool_mutex, &ts);
	}
	for (struct running_thread* current = running_threads; current != NULL; )
	{
	    int errcode = pthread_tryjoin_np(*(current->thread_id), NULL);
	    if (errcode == 0)
	    {
		// get it out of the list
		if (last == NULL)
		{
		    // first in the list
		    running_threads = current->next;
		    FREE(current);
		    current = running_threads;
		}
		else
		{
		    // have to get it out of the list...
		    struct running_thread* tmp = current;
		    last->next = current->next;
		    current = current->next;
		    FREE(tmp);
		}
		curr_num_threads--;
	    }
	    else
	    {
		last = current;
		current = current->next;
	    }
	}
	pthread_mutex_unlock(&tpool_mutex);
#else
	threads[i].threadfunc(threads[i].thread_arg);
	i++;
#endif
    }
#if HAVE_LIBPTHREAD == 1
    // wait for all threads to finish
    while (running_threads != NULL)
    {
	pthread_join(*(running_threads->thread_id), NULL);
	struct running_thread* tmp = running_threads;
	running_threads = running_threads->next;
	FREE(tmp);
    }
#endif
}

void cleanup_fdlpenv_plans()
{
    for (int i = 0; i < num_fdlpenv_plans; i++)
    {
	if (fdlpenv_plans != NULL)
	{
	    if (fdlpenv_plans[i] != NULL)
	    {
#if HAVE_LIBPTHREAD == 1
		pthread_mutex_lock(&fftw_mutex);
#endif
		fftw_destroy_plan(fdlpenv_plans[i]);
#if HAVE_LIBPTHREAD == 1
		pthread_mutex_unlock(&fftw_mutex);
#endif
	    }
	}
	if (fdlpenv_input_buffers != NULL)
	{
	    if (fdlpenv_input_buffers[i] != NULL)
	    {
		FREE(fdlpenv_input_buffers[i]);
		fdlpenv_input_buffers[i] = NULL;
	    }
	}
	if (fdlpenv_output_buffers != NULL)
	{
	    if (fdlpenv_output_buffers[i] != NULL)
	    {
		FREE(fdlpenv_output_buffers[i]);
		fdlpenv_output_buffers[i] = NULL;
	    }
	}
    }
    if (fdlpenv_plans != NULL)
    {
	FREE(fdlpenv_plans);
	fdlpenv_plans = NULL;
    }
    if (fdlpenv_plan_sizes != NULL)
    {
	FREE(fdlpenv_plan_sizes);
	fdlpenv_plan_sizes = NULL;
    }
    if (fdlpenv_input_buffers != NULL)
    {
	FREE(fdlpenv_input_buffers);
	fdlpenv_input_buffers = NULL;
    }
    if (fdlpenv_output_buffers != NULL)
    {
	FREE(fdlpenv_output_buffers);
	fdlpenv_output_buffers = NULL;
    }
    num_fdlpenv_plans = 0;
}

void cleanup_lpc_plans()
{
    for (int i = 0; i < num_lpc_plans; i++)
    {
	if (lpc_r2c_plans != NULL)
	{
	    if (lpc_r2c_plans[i] != NULL)
	    {
#if HAVE_LIBPTHREAD == 1
		pthread_mutex_lock(&fftw_mutex);
#endif
		fftw_destroy_plan(lpc_r2c_plans[i]);
#if HAVE_LIBPTHREAD == 1
		pthread_mutex_unlock(&fftw_mutex);
#endif
	    }
	}
	if (lpc_r2c_input_buffers != NULL)
	{
	    if (lpc_r2c_input_buffers[i] != NULL)
	    {
		FREE(lpc_r2c_input_buffers[i]);
		lpc_r2c_input_buffers[i] = NULL;
	    }
	}
	if (lpc_r2c_output_buffers != NULL)
	{
	    if (lpc_r2c_output_buffers[i] != NULL)
	    {
		FREE(lpc_r2c_output_buffers[i]);
		lpc_r2c_output_buffers[i] = NULL;
	    }
	}
	if (lpc_c2r_plans != NULL)
	{
	    if (lpc_c2r_plans[i] != NULL)
	    {
#if HAVE_LIBPTHREAD == 1
		pthread_mutex_lock(&fftw_mutex);
#endif
		fftw_destroy_plan(lpc_c2r_plans[i]);
#if HAVE_LIBPTHREAD == 1
		pthread_mutex_unlock(&fftw_mutex);
#endif
	    }
	}
	if (lpc_c2r_input_buffers != NULL)
	{
	    if (lpc_c2r_input_buffers[i] != NULL)
	    {
		FREE(lpc_c2r_input_buffers[i]);
		lpc_c2r_input_buffers[i] = NULL;
	    }
	}
	if (lpc_c2r_output_buffers != NULL)
	{
	    if (lpc_c2r_output_buffers[i] != NULL)
	    {
		FREE(lpc_c2r_output_buffers[i]);
		lpc_c2r_output_buffers[i] = NULL;
	    }
	}
    }
    if (lpc_r2c_plans != NULL)
    {
	FREE(lpc_r2c_plans);
	lpc_r2c_plans = NULL;
    }
    if (lpc_r2c_plan_sizes != NULL)
    {
	FREE(lpc_r2c_plan_sizes);
	lpc_r2c_plan_sizes = NULL;
    }
    if (lpc_r2c_input_buffers != NULL)
    {
	FREE(lpc_r2c_input_buffers);
	lpc_r2c_input_buffers = NULL;
    }
    if (lpc_r2c_output_buffers != NULL)
    {
	FREE(lpc_r2c_output_buffers);
	lpc_r2c_output_buffers = NULL;
    }
    if (lpc_c2r_plans != NULL)
    {
	FREE(lpc_c2r_plans);
	lpc_c2r_plans = NULL;
    }
    if (lpc_c2r_plan_sizes != NULL)
    {
	FREE(lpc_c2r_plan_sizes);
	lpc_c2r_plan_sizes = NULL;
    }
    if (lpc_c2r_input_buffers != NULL)
    {
	FREE(lpc_c2r_input_buffers);
	lpc_c2r_input_buffers = NULL;
    }
    if (lpc_c2r_output_buffers != NULL)
    {
	FREE(lpc_c2r_output_buffers);
	lpc_c2r_output_buffers = NULL;
    }
    num_lpc_plans = 0;
}

void cleanup_fftw_plans()
{
    if (dct_plan != NULL)
    {
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_lock(&fftw_mutex);
#endif
	fftw_destroy_plan(dct_plan);
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_unlock(&fftw_mutex);
#endif
    }
    if (dct_buffer != NULL)
    {
	FREE(dct_buffer);
	dct_buffer = NULL;
    }
    cleanup_lpc_plans();
    cleanup_fdlpenv_plans();
}

void usage();

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
	usage();
    }

    if (endptr == str) {
	fprintf(stderr, "No digits were found when parsing option %s\n", argname);
	usage();
    }

    if (*endptr != '\0')
    {
	fprintf(stderr, "Excess characters after integer argument of option %s: %s\n", argname, endptr);
	usage();
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
	usage();
    }

    if (endptr == str) {
	fprintf(stderr, "No digits were found when parsing option %s\n", argname);
	usage();
    }

    if (*endptr != '\0')
    {
	fprintf(stderr, "Excess characters after float argument of option %s: %s\n", argname, endptr);
	usage();
    }

    return val;
}

void usage()
{
    fatal("\nFDLP Feature Extraction software\n"
	    "USAGE:\n"
	    "cfdlp [options] -i <str> [-o <str> | -print <str>]\n"

	    "\nOPTIONS\n\n"
	    " -h, --help\t\tPrint this help and exit\n"
	    " -v, --verbose\t\tPrint out the computation time for every fdlp window\n"
	    "\nIO options:\n\n"
	    " -i <str>\t\tInput file name. Only signed 16-bit little endian raw files are supported. REQUIRED\n"
	    " -o <str>\t\tOutput file name for raw binary float output. Either this or -print is REQUIRED\n"
	    " -print <str>\t\tOutput file name for ascii output, one frame per line. Either this or -o is REQUIRED\n"
	    " -so <str>\t\tOutput file name for spectrogram output in addition to normal features (do NOT use -specgram with this option)\n"
	    "\t\t\t\tPlease be aware that, if wiener filtering is used, the alpha parameter affects the spectrum and the -specgram flag internally sets the feature type to 1 (default alpha 0.1)\n"
	    " -sr <str>\t\tInput samplerate in Hertz. Only 8000 and 16000 Hz are supported. (8000)\n"

	    "\nWindowing options:\n\n"
	    " -fdplpwin <sec>\tLength of FDPLP window in sec as a float (better for reverberant environments when gain normalization is used: 10) (5)\n"
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
	    " -downsampling-factor <int>\tBy which factor to downsample the fdlp envelope. Considerably speeds up computation. (1)\n"
	    "\t\t\t\tA factor of 20 was found to be useful for 8kHz data.\n"
	    " -downsampling-sr <int>\tWhat sample rate to downsample the fdlp envelope to. (-sr/-downsampling-factor)\n"
	    "\t\t\t\tEssentially, this is an alternate way of giving -downsampling-factor independent of the sampling rate.\n"
	    "\t\t\t\tIf both are given and do not match, -downsampling-factor takes precedence.\n"

	    "\nAdditive noise suppression options:\n\n"
	    " -apply-wiener <flag>\tApply Wiener filter (helps against additive noise) (0)\n"
	    " -wiener-alpha <float>\tsets the parameter alpha of the wiener filter (0.9 for modulation and 0.1 for spectral features)\n"
	    " -vadfile <str>\t\tname of the VAD file to read in. Has to be ascii, one char per frame (not given -> energy-based ad-hoc VAD)\n"
	    " -speechchar <char>\tthe char representing speech in the VAD file ('1')\n"
	    " -nonspeechchar <char>\tthe char representing non-speech in the VAD file ('0')\n"
	    " -vad-grace <int>\tmaximum difference between number of frames in VAD file compared to how many are computed. If there are less frames in the VAD file, the last VAD label gets repeated. (2)\n"

#if HAVE_LIBPTHREAD == 1
	    "\nMultithreading options:\n\n"
	    " -max-threads <int>:\tMaximum number of threads to use. -1 means no limit, i.e. as many threads as there are bands (1)\n"
#endif
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
	else if ( strcmp(argv[i], "-v") == 0
		|| strcmp(argv[i], "--verbose") == 0
		|| strcmp(argv[i], "-verbose") == 0)
	{
	    verbose = 1;
	}
	else if ( strcmp(argv[i], "-i") == 0 )
	{
	    if (i < argc - 1 && argv[i+1][0] != '-')
	    {
		infile = argv[++i];
	    }
	    else
	    {
		fprintf(stderr, "No input file given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-o") == 0 )
	{
	    if (i < argc - 1 && argv[i+1][0] != '-')
	    {
		outfile = argv[++i];
	    }
	    else
	    {
		fprintf(stderr, "No output file given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-print") == 0 )
	{
	    if (i < argc - 1 && argv[i+1][0] != '-')
	    {
		printfile = argv[++i];
	    }
	    else
	    {
		fprintf(stderr, "No print output file given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-so") == 0 )
	{
	    if (i < argc - 1 && argv[i+1][0] != '-')
	    {
		specfile = argv[++i];
	    }
	    else
	    {
		fprintf(stderr, "No spectrogram output file given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-sr") == 0 )
	{
	    if (i < argc - 1)
	    {
		Fs = str_to_int(argv[++i], "-sr");
		if (Fs != 8000 && Fs != 16000) {
		    fatal("Unsupported sample rate! Only 8000 and 16000 are supported.");
		}
	    }
	    else
	    {
		fprintf(stderr, "No sample rate given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-gn") == 0 )
	{
	    if (i < argc - 1)
	    {
		do_gain_norm = str_to_int(argv[++i], "-gn");
	    }
	    else
	    {
		fprintf(stderr, "No gain normalization flag given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-spec") == 0 
		|| strcmp(argv[i], "-feat") == 0
		)
	{
	    if (i < argc - 1)
	    {
		do_spec = str_to_int(argv[++i], "-feat");
		if (do_spec != 0 && do_spec != 1)
		{
		    fprintf(stderr, "Error: -feat: Unsupported feature type!\n");
		    usage();
		}
	    }
	    else
	    {
		fprintf(stderr, "No feature type given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-axis") == 0 )
	{
	    if (i < argc - 1)
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
	    else
	    {
		fprintf(stderr, "No axis given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-specgram") == 0 )
	{
	    if (i < argc - 1)
	    {
		specgrm = str_to_int(argv[++i], "-specgram");
	    }
	    else
	    {
		fprintf(stderr, "No spectrogram flag given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-limit-range") == 0 )
	{
	    if (i < argc - 1)
	    {
		limit_range = str_to_int(argv[++i], "-limit-range");
	    }
	    else
	    {
		fprintf(stderr, "No limit range flag given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-apply-wiener") == 0 )
	{
	    if (i < argc - 1)
	    {
		do_wiener = str_to_int(argv[++i], "-apply-wiener");
	    }
	    else
	    {
		fprintf(stderr, "No apply-wiener flag given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-wiener-alpha") == 0 )
	{
	    if (i < argc - 1)
	    {
		wiener_alpha = str_to_float(argv[++i], "-wiener-alpha");
		wiener_alpha_given = 1;
	    }
	    else
	    {
		fprintf(stderr, "No wiener-alpha given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-fdplpwin") == 0)
	{
	    if (i < argc - 1)
	    {
		fdplp_win_len_sec = str_to_float(argv[++i], "-fdplpwin");
	    }
	    else
	    {
		fprintf(stderr, "No fdplp window length given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-truncate-last") == 0)
	{
	    if (i < argc - 1)
	    {
		truncate_last = str_to_int(argv[++i], "-truncate-last");
	    }
	    else
	    {
		fprintf(stderr, "No truncate-last flag given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-skip-bands") == 0)
	{
	    if (i < argc - 1)
	    {
		skip_bands = str_to_int(argv[++i], "-skip-bands");
	    }
	    else
	    {
		fprintf(stderr, "No skip-bands flag given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-vadfile") == 0)
	{
	    if (i < argc - 1 && argv[i+1][0] != '-')
	    {
		vadfile = argv[++i];
	    }
	    else
	    {
		fprintf(stderr, "No vad file given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-speechchar") == 0)
	{
	    if (i < argc - 1)
	    {
		speechchar = argv[++i][0];
	    }
	    else
	    {
		fprintf(stderr, "No speech char given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-nonspeechchar") == 0)
	{
	    if (i < argc - 1)
	    {
		nspeechchar = argv[++i][0];
	    }
	    else
	    {
		fprintf(stderr, "No nonspeech char given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-vad-grace") == 0)
	{
	    if (i < argc - 1)
	    {
		vad_grace = str_to_int(argv[++i], "-vad-grace");
	    }
	    else
	    {
		fprintf(stderr, "No vad grace given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-nceps") == 0)
	{
	    if (i < argc - 1)
	    {
		num_cepstral_coeffs = str_to_int(argv[++i], "-nceps");
	    }
	    else
	    {
		fprintf(stderr, "No nceps given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-shortterm-mode") == 0)
	{
	    if (i < argc - 1)
	    {
		int shortterm_mode = str_to_int(argv[++i], "-shortterm-mode");
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
	    else
	    {
		fprintf(stderr, "No shortterm mode given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-modulation-mode") == 0)
	{
	    if (i < argc - 1)
	    {
		int modmode = str_to_int(argv[++i], "-modulation-mode");
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
		fprintf(stderr, "No modulation mode given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-downsampling-factor") == 0 )
	{
	    if (i < argc - 1)
	    {
		factor = str_to_int(argv[++i], "-downsampling-factor");
	    }
	    else
	    {
		fprintf(stderr, "No downsampling factor given!\n");
		usage();
	    }
	}
	else if ( strcmp(argv[i], "-downsampling-sr") == 0 )
	{
	    if (i < argc - 1)
	    {
		Fs1 = str_to_int(argv[++i], "-downsampling-sr");
	    }
	    else
	    {
		fprintf(stderr, "No downsampling sample rate given!\n");
		usage();
	    }
	}
#if HAVE_LIBPTHREAD == 1
	else if ( strcmp(argv[i], "-max-threads") == 0 )
	{
	    if (i < argc - 1)
	    {
		max_num_threads = str_to_int(argv[++i], "-max-threads");
	    }
	    else
	    {
		fprintf(stderr, "No maximum number of threads given!\n");
		usage();
	    }
	}
#endif
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

    if (specgrm)
    {
	do_spec = 1;
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
    if (factor < 0 && Fs1 > 0)
    {
	factor = Fs / Fs1;
    }
    else if (factor > 0)
    {
	Fs1 = Fs / factor;
    }
    else
    {
	factor = 1;
	Fs1 = Fs;
    }

    if (fdplp_win_len_sec <= 0.) {
	fprintf(stderr, "FDLP window can not have a negative length!\n");
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
	if (E[i] < 0.) {
	    fprintf(stderr, "E[%d]=%g, k[%d]=%g, k[i]^2=%g, E[%d-1]=%g\n",
		    i, E[i], i, k[i], k[i]*k[i], i, E[i-1]);
	    fatal("E[i] negative -> something must be wrong with the data.");
	}
    }

    // Copy final iteration coeffs to output array
    g = sqrt(E[p]);
    // nantest
    for ( int i = 0; i <= p; i++ )
    {
	poles[i] = alpha[p*(p+1)+i];
	if (!do_gain_norm)
	{
	    if (E[p] <= 0.) {
		// g is nan now (negative) or 0 (if E[p] == 0.) -> complex
		// squareroot or division by zero. all boils down to all-zero
		// poles
		poles[i] = 0.;
	    }
	    else
	    {
		poles[i] /= g; 
	    }
	}	
    }
    FREE(k);
    FREE(E);
    FREE(alpha);
}

void lpc( double *y, int len, int order, int compr, float *poles, int myband ) 
{
    // Compute autocorrelation vector or matrix
    int N = pow(2,ceil(log2(2*len-1)));

    if (N != lpc_r2c_plan_sizes[myband])
    {
	lpc_r2c_plan_sizes[myband] = N;
	if (lpc_r2c_plans[myband] != NULL)
	{
#if HAVE_LIBPTHREAD == 1
	    pthread_mutex_lock(&fftw_mutex);
#endif
	    fftw_destroy_plan(lpc_r2c_plans[myband]);
#if HAVE_LIBPTHREAD == 1
	    pthread_mutex_unlock(&fftw_mutex);
#endif
	    lpc_r2c_plans[myband] = NULL;
	}
	if (lpc_r2c_input_buffers[myband] != NULL)
	{
	    FREE(lpc_r2c_input_buffers[myband]);
	    lpc_r2c_input_buffers[myband] = NULL;
	}
	if (lpc_r2c_output_buffers[myband] != NULL)
	{
	    FREE(lpc_r2c_output_buffers[myband]);
	    lpc_r2c_output_buffers[myband] = NULL;
	}
	if (lpc_c2r_plans[myband] != NULL)
	{
	    FREE(lpc_c2r_plans[myband]);
	    lpc_c2r_plans[myband] = NULL;
	}
	if (lpc_c2r_output_buffers[myband] != NULL)
	{
	    FREE(lpc_c2r_output_buffers[myband]);
	    lpc_c2r_output_buffers[myband] = NULL;
	}
    }

    if (lpc_r2c_input_buffers[myband] == NULL)
    {
	lpc_r2c_input_buffers[myband] = (double*) MALLOC(lpc_r2c_plan_sizes[myband] * sizeof(double));
    }

    for ( int n = 0; n < N; n++ )
    {
	if ( n < len )
	{
	    lpc_r2c_input_buffers[myband][n] = y[n];
	}
	else
	{
	    lpc_r2c_input_buffers[myband][n] = 0;
	}
    }   

    if (lpc_r2c_output_buffers[myband] == NULL)
    {
	lpc_r2c_output_buffers[myband] = (complex*) MALLOC(lpc_r2c_plan_sizes[myband] * sizeof(complex));
	memset(lpc_r2c_output_buffers[myband], 0, N * sizeof(complex)); // fix uninitialized value-issue in multiplication below 
    }

    if (lpc_r2c_plans[myband] == NULL)
    {
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_lock(&fftw_mutex);
#endif
	lpc_r2c_plans[myband] = fftw_plan_dft_r2c_1d(
		lpc_r2c_plan_sizes[myband],
		lpc_r2c_input_buffers[myband],
		lpc_r2c_output_buffers[myband],
		FFTW_ESTIMATE);
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_unlock(&fftw_mutex);
#endif
    }
    fftw_execute(lpc_r2c_plans[myband]);

    for ( int n = 0; n < N; n++ )
    {
	lpc_r2c_output_buffers[myband][n] = lpc_r2c_output_buffers[myband][n]
	    *conj(lpc_r2c_output_buffers[myband][n])
	    /len; //add compr
    }

    if (lpc_c2r_output_buffers[myband] == NULL)
    {
	lpc_c2r_output_buffers[myband] = (double*) MALLOC(lpc_r2c_plan_sizes[myband] * sizeof(double));
    }
    if (lpc_c2r_plans[myband] == NULL)
    {
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_lock(&fftw_mutex);
#endif
	lpc_c2r_plans[myband] = fftw_plan_dft_c2r_1d(
		lpc_r2c_plan_sizes[myband],
		lpc_r2c_output_buffers[myband],
		lpc_c2r_output_buffers[myband],
		FFTW_ESTIMATE);
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_unlock(&fftw_mutex);
#endif
    }
    fftw_execute(lpc_c2r_plans[myband]);
    for ( int n = 0; n < N; n++ )
    {
	lpc_c2r_output_buffers[myband][n] /= N;
    }

    levinson(order, lpc_c2r_output_buffers[myband], poles);
}

void hlpc_wiener(double *y, int len, int order, float *poles, int orig_len, int *vadindices, int Nindices, int myband)
{
    int wlen = round(DEFAULT_SHORTTERM_WINLEN_MS* Fs);
    float SP = DEFAULT_SHORTTERM_SHIFT_PERCENTAGE;

    int N = 2 * orig_len - 1;

    if (N != lpc_r2c_plan_sizes[myband])
    {
	lpc_r2c_plan_sizes[myband] = N;
	if (lpc_r2c_plans[myband] != NULL)
	{
#if HAVE_LIBPTHREAD == 1
	    pthread_mutex_lock(&fftw_mutex);
#endif
	    fftw_destroy_plan(lpc_r2c_plans[myband]);
#if HAVE_LIBPTHREAD == 1
	    pthread_mutex_unlock(&fftw_mutex);
#endif
	    lpc_r2c_plans[myband] = NULL;
	}
	if (lpc_r2c_input_buffers[myband] != NULL)
	{
	    FREE(lpc_r2c_input_buffers[myband]);
	    lpc_r2c_input_buffers[myband] = NULL;
	}
	if (lpc_r2c_output_buffers[myband] != NULL)
	{
	    FREE(lpc_r2c_output_buffers[myband]);
	    lpc_r2c_output_buffers[myband] = NULL;
	}
    }
    if (lpc_r2c_input_buffers[myband] == NULL)
    {
	lpc_r2c_input_buffers[myband] = (double*) MALLOC(N * sizeof(double));
    }
    for (int n = 0; n < N; n++)
    {
	if (n < len)
	{
	    lpc_r2c_input_buffers[myband][n] = y[n];
	}
	else
	{
	    lpc_r2c_input_buffers[myband][n] = 0.;
	}
    }
    if (lpc_r2c_output_buffers[myband] == NULL)
    {
	lpc_r2c_output_buffers[myband] = (complex*) MALLOC(N * sizeof(complex));
	memset(lpc_r2c_output_buffers[myband], 0, N * sizeof(complex));
    }
    if (lpc_r2c_plans[myband] == NULL)
    {
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_lock(&fftw_mutex);
#endif
	lpc_r2c_plans[myband] = fftw_plan_dft_r2c_1d(
		lpc_r2c_plan_sizes[myband],
		lpc_r2c_input_buffers[myband],
		lpc_r2c_output_buffers[myband],
		FFTW_ESTIMATE);
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_unlock(&fftw_mutex);
#endif
    }
    fftw_execute(lpc_r2c_plans[myband]);

    float *ENV = (float *) MALLOC(orig_len * sizeof(float));
    for (int i = 0; i < orig_len; i++)
    {
	ENV[i] = (float)cabs(lpc_r2c_output_buffers[myband][i])
	    * (float)cabs(lpc_r2c_output_buffers[myband][i]);
    }

    int envlen = orig_len;
    int envframes = 0;
    int overlap = wlen - (int)round((float)wlen * SP);
    float *fftframes = fconstruct_frames_wiener(&ENV, &envlen, wlen, overlap, &envframes);

    float *X = (float *)MALLOC(envframes * wlen * sizeof(float));
    float *Pn = (float *)MALLOC(wlen * sizeof(float));
    pthread_mutex_lock(&fftw_mutex);
    if (Pn_buf == NULL)
    {
	Pn_buf = (float*) MALLOC(nbands * wlen * sizeof(float));
	Pn_buf_valid = 0;
    }
    pthread_mutex_unlock(&fftw_mutex);
    for (int i = 0; i < wlen; i++) {
	if (Nindices == 0)
	{
	    if (Pn_buf_valid == 1)
	    {
		Pn[i] = Pn_buf[myband * wlen + i];
	    }
	    else
	    {
		Pn[i] = 1.; // neutral for the division lateron
	    }
	}
	else
	{
	    Pn[i] = 0;
	}
    }
    if (Nindices > 0)
    {
	for (int i = 0; i < Nindices; i++) {
	    int frameindex = vadindices[i];
	    for (int j = 0; j < wlen; j++) {
		Pn[j] += fftframes[frameindex * wlen + j];
	    }
	}
	for (int i = 0; i < wlen; i++) {
	    Pn[i] /= Nindices;
	    Pn_buf[myband * wlen + i] = Pn[i];
	}
	Pn_buf_valid = 1;
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
    if (lpc_c2r_plan_sizes[myband] != N)
    {
	lpc_c2r_plan_sizes[myband] = N;
	if (lpc_c2r_plans[myband] != NULL)
	{
#if HAVE_LIBPTHREAD == 1
	    pthread_mutex_lock(&fftw_mutex);
#endif
	    fftw_destroy_plan(lpc_c2r_plans[myband]);
#if HAVE_LIBPTHREAD == 1
	    pthread_mutex_unlock(&fftw_mutex);
#endif
	    lpc_c2r_plans[myband] = NULL;
	}
	if (lpc_c2r_input_buffers[myband] != NULL)
	{
	    FREE(lpc_c2r_input_buffers[myband]);
	    lpc_c2r_input_buffers[myband] = NULL;
	}
	if (lpc_c2r_output_buffers[myband] != NULL)
	{
	    FREE(lpc_c2r_output_buffers[myband]);
	    lpc_c2r_output_buffers[myband] = NULL;
	}
    }
    if (lpc_c2r_input_buffers[myband] == NULL)
    {
	lpc_c2r_input_buffers[myband] = (complex*) MALLOC(N * sizeof(complex));
    }
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
	lpc_c2r_input_buffers[myband][i] = ENV_output[env_output_index] / len;
    }

    if (lpc_c2r_output_buffers[myband] == NULL)
    {
	lpc_c2r_output_buffers[myband] = (double*) MALLOC(N * sizeof(double));
    }

    if (lpc_c2r_plans[myband] == NULL)
    {
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_lock(&fftw_mutex);
#endif
	lpc_c2r_plans[myband] = fftw_plan_dft_c2r_1d(
		lpc_c2r_plan_sizes[myband],
		lpc_c2r_input_buffers[myband],
		lpc_c2r_output_buffers[myband],
		FFTW_ESTIMATE);
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_unlock(&fftw_mutex);
#endif
    }
    fftw_execute(lpc_c2r_plans[myband]);

    for ( int n = 0; n < N; n++ )
    {
	lpc_c2r_output_buffers[myband][n] /= N;
    }

    levinson(order, lpc_c2r_output_buffers[myband], poles);

    FREE(ENV);
    FREE(fftframes);
    FREE(Pn);
    FREE(X);
    FREE(ENV_output);
    FREE(inv_win);
}

void* lpc_pthread_wrapper(void* arg)
{
    struct lpc_info* info = (struct lpc_info*)arg;
    if (do_wiener)
    {
	hlpc_wiener(info->y, info->len, info->order, info->poles, info->orig_len, info->vadindices, info->Nindices, info->band);
    }
    else
    {
	lpc(info->y, info->len, info->order, info->compression, info->poles, info->band);
    }
    pthread_cond_signal(&tpool_cond);
    return NULL;
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
    if (dct_plan_size != N)
    {
	if (dct_plan != NULL)
	{
#if HAVE_LIBPTHREAD == 1
	    pthread_mutex_lock(&fftw_mutex);
#endif
	    fftw_destroy_plan(dct_plan);
#if HAVE_LIBPTHREAD == 1
	    pthread_mutex_unlock(&fftw_mutex);
#endif
	    dct_plan = NULL;
	}
	if (dct_buffer != NULL)
	{
	    FREE(dct_buffer);
	    dct_buffer = NULL;
	}
	dct_plan_size = N;
	dct_buffer = (double *) MALLOC(dct_plan_size * sizeof(double));
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_lock(&fftw_mutex);
#endif
	dct_plan = fftw_plan_r2r_1d(dct_plan_size, dct_buffer, dct_buffer, FFTW_REDFT10, FFTW_ESTIMATE);
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_unlock(&fftw_mutex);
#endif
    }

    for ( int n = 0; n < N; n++ )
    {
	dct_buffer[n] = (double) x[n];
    }

    fftw_execute(dct_plan);

    for ( int n = 0; n < N; n++ )
    {
        dct_buffer[n] /= sqrt(2.0*N);
    }
    dct_buffer[0] /= sqrt(2);

    int fdlpwin = N;

    double *y = dct_buffer;
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
	if  (verbose) fprintf(stderr, "(Re)creating auditory filter bank (nbands or fdlpwin changed)\n");
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

    if (nbands != bank_nbands)
    {
	cleanup_lpc_plans();
	cleanup_fdlpenv_plans();
	if (band_threads != NULL)
	{
	    FREE(band_threads);
	    band_threads = NULL;
	}
    }
    
    nbands = bank_nbands;

    if (lpc_r2c_plans == NULL)
    {
	lpc_r2c_plans = (fftw_plan*) MALLOC(nbands * sizeof(fftw_plan));
	lpc_r2c_plan_sizes = (int*) MALLOC(nbands * sizeof(int));
	lpc_r2c_input_buffers = (double**) MALLOC(nbands * sizeof(double*));
	lpc_r2c_output_buffers = (complex**) MALLOC(nbands * sizeof(complex*));
	lpc_c2r_plans = (fftw_plan*) MALLOC(nbands * sizeof(fftw_plan));
	lpc_c2r_plan_sizes = (int*) MALLOC(nbands * sizeof(int));
	lpc_c2r_input_buffers = (complex **) MALLOC(nbands * sizeof(complex*));
	lpc_c2r_output_buffers = (double**) MALLOC(nbands * sizeof(double*));
	num_lpc_plans = nbands;
	for (int i = 0; i < nbands; i++)
	{
	    lpc_r2c_plans[i] = NULL;
	    lpc_r2c_plan_sizes[i] = -1;
	    lpc_r2c_input_buffers[i] = NULL;
	    lpc_r2c_output_buffers[i] = NULL;
	    lpc_c2r_plans[i] = NULL;
	    lpc_c2r_plan_sizes[i] = -1;
	    lpc_c2r_input_buffers[i] = NULL;
	    lpc_c2r_output_buffers[i] = NULL;
	}
    }
    if (fdlpenv_plans == NULL)
    {
	fdlpenv_plans = (fftw_plan*) MALLOC(nbands * sizeof(fftw_plan));
	fdlpenv_plan_sizes = (int*) MALLOC(nbands * sizeof(int));
	fdlpenv_input_buffers = (double**) MALLOC(nbands * sizeof(double*));
	fdlpenv_output_buffers = (complex**) MALLOC(nbands * sizeof(complex*));
	num_fdlpenv_plans = nbands;
	for (int i = 0; i < nbands; i++)
	{
	    fdlpenv_plans[i] = NULL;
	    fdlpenv_plan_sizes[i] = -1;
	    fdlpenv_input_buffers[i] = NULL;
	    fdlpenv_output_buffers[i] = NULL;
	}
    }

    if (band_threads == NULL)
    {
	band_threads = (struct thread_info*) MALLOC(nbands * sizeof(struct thread_info));
    }
    
    if (verbose) fprintf(stderr, "Number of sub-bands = %d\n", nbands);	
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
    double **y_filt = (double**) MALLOC(nbands * sizeof(double*));
    struct lpc_info* info = (struct lpc_info*) MALLOC(nbands * sizeof(struct lpc_info));
    for ( int i = 0; i < nbands; i++ )
    {
	y_filt[i] = (double*) MALLOC(fdlpwin * sizeof(double));
	int Nsub = indices[2*i+1]-indices[2*i]+1;
	for ( int n = 0; n < fdlpwin; n++ )
	{
	    if ( n < Nsub )
	    {
		y_filt[i][n] = y[indices[2*i]+n] * wts[i*fdlpwin+indices[2*i]+n];
	    }
	    else
	    {
		y_filt[i][n] = 0;
	    }
	}
	info[i].y = y_filt[i];
	info[i].len = Nsub;
	info[i].order = *Np;
	info[i].compression = 1;
	info[i].poles = p + i * (*Np+1);
	info[i].orig_len = N;
	info[i].vadindices = NIS;
	info[i].Nindices = NNIS;
	info[i].band = i;

	band_threads[i].threadfunc = lpc_pthread_wrapper;
	band_threads[i].thread_arg = (void*)&info[i];
	//pthread_create(&band_threads[i], NULL, lpc_pthread_wrapper, (void*)&info[i]);
    }

    run_threads(band_threads, nbands);

    for (int i = 0; i < nbands; i++)
    {
	//pthread_join(band_threads[i], NULL);
	FREE(y_filt[i]);
    }

    FREE(y_filt);
    FREE(info);
    if (NIS != NULL)
    {
	FREE(NIS);
    }

    return p;
}

float * fdlpenv( float *p, int Np, int N, int myband )
{
    float *env = (float *) MALLOC( N*sizeof(float) );

    int nfft = 2 * (MAX(Np, N) - 1); // --> N = nfft / 2 + 1 == half (fft is symmetric)
    if (nfft != fdlpenv_plan_sizes[myband])
    {
	fdlpenv_plan_sizes[myband] = nfft;
	if (fdlpenv_plans[myband] != NULL)
	{
#if HAVE_LIBPTHREAD == 1
	    pthread_mutex_lock(&fftw_mutex);
#endif
	    fftw_destroy_plan(fdlpenv_plans[myband]);
#if HAVE_LIBPTHREAD == 1
	    pthread_mutex_unlock(&fftw_mutex);
#endif
	    fdlpenv_plans[myband] = NULL;
	}
	if (fdlpenv_input_buffers[myband] != NULL)
	{
	    FREE(fdlpenv_input_buffers[myband]);
	    fdlpenv_input_buffers[myband] = NULL;
	}
	if (fdlpenv_output_buffers[myband] != NULL)
	{
	    FREE(fdlpenv_output_buffers[myband]);
	    fdlpenv_output_buffers[myband] = NULL;
	}
    }

    if (fdlpenv_input_buffers[myband] == NULL)
    {
	fdlpenv_input_buffers[myband] = (double*) MALLOC(fdlpenv_plan_sizes[myband] * sizeof(double));
    }
    for ( int n = 0; n < nfft; n++ )
    {
	if ( n <= Np )
	{
	    fdlpenv_input_buffers[myband][n] = p[n];
	}
	else
	{
	    fdlpenv_input_buffers[myband][n] = 0;
	}
    }   

    if (fdlpenv_output_buffers[myband] == NULL)
    {
	fdlpenv_output_buffers[myband] = (complex*) MALLOC(fdlpenv_plan_sizes[myband] * sizeof(complex));
    }

    if (fdlpenv_plans[myband] == NULL)
    {
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_lock(&fftw_mutex);
#endif
	fdlpenv_plans[myband] = fftw_plan_dft_r2c_1d(
		fdlpenv_plan_sizes[myband],
		fdlpenv_input_buffers[myband],
		fdlpenv_output_buffers[myband],
		FFTW_ESTIMATE);
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_unlock(&fftw_mutex);
#endif
    }
    fftw_execute(fdlpenv_plans[myband]);

    for ( int n = 0; n < N; n++ )
    {
	fdlpenv_output_buffers[myband][n] = 1.0/fdlpenv_output_buffers[myband][n];
	env[n] = 2*fdlpenv_output_buffers[myband][n]*conj(fdlpenv_output_buffers[myband][n]);
    }

    return env;
}

float * fdlpenv_mod( float *p, int Np, int N, int myband )
{
    float *env = (float *) MALLOC( N*sizeof(float) );

    int nfft = pow(2,ceil(log2(N))+1);
    if (nfft != fdlpenv_plan_sizes[myband])
    {
	fdlpenv_plan_sizes[myband] = nfft;
	if (fdlpenv_plans[myband] != NULL)
	{
#if HAVE_LIBPTHREAD == 1
	    pthread_mutex_lock(&fftw_mutex);
#endif
	    fftw_destroy_plan(fdlpenv_plans[myband]);
#if HAVE_LIBPTHREAD == 1
	    pthread_mutex_unlock(&fftw_mutex);
#endif
	    fdlpenv_plans[myband] = NULL;
	}
	if (fdlpenv_input_buffers[myband] != NULL)
	{
	    FREE(fdlpenv_input_buffers[myband]);
	    fdlpenv_input_buffers[myband] = NULL;
	}
	if (fdlpenv_output_buffers[myband] != NULL)
	{
	    FREE(fdlpenv_output_buffers[myband]);
	    fdlpenv_output_buffers[myband] = NULL;
	}
    }

    if (fdlpenv_input_buffers[myband] == NULL)
    {
	fdlpenv_input_buffers[myband] = (double*) MALLOC(fdlpenv_plan_sizes[myband] * sizeof(double));
    }
    for ( int n = 0; n < nfft; n++ )
    {
	if ( n <= Np )
	{
	    fdlpenv_input_buffers[myband][n] = p[n];
	}
	else
	{
	    fdlpenv_input_buffers[myband][n] = 0;
	}
    }   

    if (fdlpenv_output_buffers[myband] == NULL)
    {
	fdlpenv_output_buffers[myband] = (complex*)MALLOC(fdlpenv_plan_sizes[myband] * sizeof(complex));
    }
    if (fdlpenv_plans[myband] == NULL)
    {
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_lock(&fftw_mutex);
#endif
	fdlpenv_plans[myband] = fftw_plan_dft_r2c_1d(
		fdlpenv_plan_sizes[myband],
		fdlpenv_input_buffers[myband],
		fdlpenv_output_buffers[myband],
		FFTW_ESTIMATE);
#if HAVE_LIBPTHREAD == 1
	pthread_mutex_unlock(&fftw_mutex);
#endif
    }
    fftw_execute(fdlpenv_plans[myband]);

    double *h = (double *) MALLOC( nfft*sizeof(double) );
    memset(h, 0, nfft * sizeof(double));

    nfft = nfft/2+1;
    for ( int n = 0; n < nfft; n++ )
    {
	fdlpenv_output_buffers[myband][n] = 1.0/fdlpenv_output_buffers[myband][n];
	h[n] = 2*fdlpenv_output_buffers[myband][n]*conj(fdlpenv_output_buffers[myband][n]);
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

    return env;
}

void spec2cep(float * frames, int fdlpwin, int nframes, int ncep, int nbands, int band, int offset, float *feats, int log_flag) 
{

#if HAVE_LIBPTHREAD == 1
    pthread_mutex_lock(&fftw_mutex);
#endif
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
#if HAVE_LIBPTHREAD == 1
    pthread_mutex_unlock(&fftw_mutex);
#endif

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
		    //feat[i] += log(frame[j])*dctm[i*fdlpwin+j];
		}
	    }
	}
    }
}

void spec2cep4energy(float * frames, int fdlpwin, int nframes, int ncep, float *final_feats, int log_flag)
{
    static int old_fdlpwin = 0, old_ncep = 0;
    if ( old_fdlpwin != fdlpwin || old_ncep != ncep )
    {
	if (dctm != NULL) {
	    FREE(dctm);
	}
	old_fdlpwin = fdlpwin;
	old_ncep = ncep;

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
		    //feat[i] += (0.33*icsi_log(frame[j],LOOKUP_TABLE,nbits_log))*dctm[i*fdlpwin+j]; //Cubic root compression and log
		    feat[i] += (0.33*log(frame[j]))*dctm[i*fdlpwin+j]; //Cubic root compression and log
		    if (isnan(feat[i])) {
			fprintf(stderr, "nan at spec2cep4energy with f=%d, i=%d, j=%d after adding 0.33*log(frame[j]=%g)*dctm[i*fdlpwin=%d+j]=%g\n",
				f, i, j, frame[j], fdlpwin, dctm[i*fdlpwin+j]);
		    }
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

    static int old_nfft = 0;

    if (old_nfft != nfft) {
	if (fft2decompm != NULL) {
	    FREE(fft2decompm);
	}
	old_nfft = nfft;
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
		temp += sqrt(energybands[f * (*nbands) + j]) * fft2decompm[i * nfft + j];
	    }
	    temp *= temp;
	    if (temp < 0) {
		fprintf(stderr, "new spectral element is negative?! Summed up these values:\n");
		for (int j = 0; j < *nbands; j++) {
		    fprintf(stderr, "%g * %g | ", energybands[f*(*nbands)+j], fft2decompm[i*nfft+j]);
		}
		fprintf(stderr, "\n");
	    }
	    new_bands[f * nfilts + i] = temp;
	}
    }

    FREE(energybands);
    *bands = new_bands;
    *nbands = nfilts;
}

void* fdlpenv_pthread_wrapper(void* arg)
{
    struct fdlpenv_info* info = (struct fdlpenv_info*)arg;
    float* fenv = NULL;
#if FDLPENV_WITH_INTERP == 1
    float* env = fdlpenv_mod(info->poles, info->Np, info->fdlplen, info->band);
    if (specfile != NULL && factor != 1)
    {
	fenv = fdlpenv_mod(info->poles, info->Np, info->ffdlplen, info->band);
    }
#else
    float* env = fdlpenv(info->poles, info->Np, info->fdlplen, info->band);
    if (specfile != NULL && factor != 1)
    {
	fenv = fdlpenv(info->poles, info->Np, info->ffdlplen, info->band);
    }
#endif
    int fnum = info->fnum;
    int send1 = info->send;
    int flen1 = info->flen;
    int fhop1 = info->fhop;
    int nframes = 0;
    int mirr_len = info->mirrlen;
    int fdlpwin = info->fdlpwin;
    int fdlpolap = info->fdlpolap;
    int nceps = info->nceps;
    int flen = info->fflen;
    int fhop = info->ffhop;
    int send = info->fsend;
    float* energybands = info->energybands;
    float* feats = info->feats;
    float *spectrogram = info->spectrogram;
    float* hamm = info->hamm;
    float *fhamm = info->fhamm;

    if (do_spec || specfile != NULL)
    {
	float *frames = fconstruct_frames(&env, &send1, flen1, flen1-fhop1, &nframes);
	float *fframes = NULL;
	if (fenv != NULL)
	{
	    int nframes2 = 0;
	    fframes = fconstruct_frames(&fenv, &send, flen, flen-fhop, &nframes2);
	    if (nframes2 != nframes)
		fatal("Number of frames different with and without downsampling?!");
	}
	for (int fr = 0; fr < fnum;fr++)
	{
	    float *envwind = frames+fr*flen1;
	    float temp = 0;
	    float *fenvwind = NULL;
	    float ftemp = 0;
	    if (fenv != NULL)
	    {
		fenvwind = fframes + fr * flen;
	    }
	    for (int ind =0;ind<flen1;ind++) 
	    {
		temp +=  envwind[ind]*hamm[ind];
		if (fenvwind)
		{
		    ftemp += fenvwind[ind] * fhamm[ind];
		}
	    }
	    if (fenvwind)
	    {
		for (int ind = flen1; ind < flen; ind++)
		{
		    ftemp += fenvwind[ind] * fhamm[ind];
		}
	    }

	    energybands[fr*nbands+info->band]= temp;

	    if (specgrm)
	    {
		feats[fr*nbands+info->band] = 0.33*log(temp);
	    }
	    if (specfile != NULL)
	    {
		float value = temp;
		if (fenvwind)
		{
		    value = ftemp;
		}
		spectrogram[fr*nbands+info->band] = 0.33*log(value);
	    }
	}
	FREE(frames);
	FREE(env);
	if (fenv)
	{
	    FREE(fframes);
	    FREE(fenv);
	}
    }
    else if (!do_spec)
    {
	int Npad1 = send1 + 2 * mirr_len;
	float* env_pad1 = (float*) MALLOC(Npad1*sizeof(float));

	int Npad2 = send1 + 1000/factor;
	float* env_pad2 = (float*) MALLOC(Npad2 * sizeof(float));
	float* env_log = (float*) MALLOC(Npad2 * sizeof(float));
	float* env_adpt = (float*) MALLOC(Npad2 * sizeof(float));

	for (int k =0;k<send1;k++)
	{
	    env_log[k] = icsi_log(env[k],LOOKUP_TABLE,nbits_log);     
	    //env_log[k] = log(env[k]);
	    sleep(0);	// Found out that icsi log is too fast and gives errors 
	}

	for ( int n = 0; n < Npad1; n++ )
	{
	    if ( n < mirr_len )
	    {
		env_pad1[n] = env_log[mirr_len-1-n];
	    }
	    else if ( n >= mirr_len && n < mirr_len + send1 )
	    {
		env_pad1[n] = env_log[n-mirr_len];	    
	    }
	    else
	    {
		env_pad1[n] = env_log[send1-(n-mirr_len-send1+1)];
	    }
	}

	float * frames = fconstruct_frames(&env_pad1, &Npad1, fdlpwin, fdlpolap, &nframes);

	if (longterm_do_static == 1)
	{
	    spec2cep(frames, fdlpwin, nframes, nceps, nbands, info->band, 0, feats, 1 );
	}

	FREE(frames);

	// do delta here
	float maxenv = 0;
	for ( int n = 0; n < Npad2; n++ )
	{
	    if ( n < 1000/factor )
	    {
		env_pad2[n] = env[0];
	    }
	    else
	    {
		env_pad2[n] = env[n-1000/factor];
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

	adapt_m(env_pad2,Npad2,Fs1,env_adpt);

	for ( int n = 0; n < Npad1; n++ )
	{
	    if ( n < mirr_len )
	    {
		env_pad1[n] = env_adpt[mirr_len-1-n+1000/factor];
	    }
	    else if ( n >= mirr_len && n < mirr_len + send1 )
	    {
		env_pad1[n] = env_adpt[n-mirr_len+1000/factor];	    
	    }
	    else
	    {
		env_pad1[n] = env_adpt[send1-(n-mirr_len-send1+1)+1000/factor];
	    }
	}

	frames = fconstruct_frames(&env_pad1, &Npad1, fdlpwin, fdlpolap, &nframes);

	if (longterm_do_dynamic == 1)
	{
	    spec2cep(frames, fdlpwin, nframes, nceps, nbands, info->band, nceps * longterm_do_static, feats, 1);
	}

	FREE(frames);
	FREE(env);

	FREE(env_pad1);
	FREE(env_pad2);
	FREE(env_log);
	FREE(env_adpt);
    }
    *(info->nframes) = nframes;
#if HAVE_LIBPTHREAD == 1
    pthread_cond_signal(&tpool_cond);
#endif
    return NULL;
}

void compute_fdlp_feats( float *x, int N, int Fs, int* nceps, float **feats, int nfeatfr, int numframes, int *dim, float **spectrogram)
{
    int flen= DEFAULT_SHORTTERM_WINLEN_MS * Fs;   // frame length corresponding to 25ms
    int flen1 = flen / factor;
    int fhop= DEFAULT_SHORTTERM_WINSHIFT_MS * Fs;   // frame overlap corresponding to 10ms
    int fhop1 = fhop / factor;
    int fnum = floor((N-flen)/fhop)+1;
    //int fnum = floor((N-flen)/fhop); // bug in feacalc will result in 1 frame less for 8kHzs data than for 16kHz :-/

    // What's the last sample that feacalc will consider?
    int send = (fnum-1)*fhop + flen;
    int send1 = send / factor;
    int fdlplen = send / factor;
    int trap = DEFAULT_LONGTERM_TRAP_FRAMECTX;  // 10 FRAME context duration
    int mirr_len = trap*fhop1;
    int fdlpwin = (DEFAULT_LONGTERM_WINLEN_MS * Fs+flen)/factor;  // Modulation spectrum Computation Window.
    int fdlpolap = fdlpwin - fhop/factor;

    int Np;
    float *p = fdlpfit_full_sig(x,N,Fs,&Np);

    //int Npad1 = send1+2*mirr_len;
    //float *env_pad1 = (float *) MALLOC(Npad1*sizeof(float));

    //int Npad2 = send1+1000/factor;
    //float *env_pad2 = (float *) MALLOC(Npad2*sizeof(float));
    //float *env_log = (float *) MALLOC(Npad2*sizeof(float));	
    //float *env_adpt = (float *) MALLOC(Npad2*sizeof(float));

    //int nframes;
    float *hamm = hamming(flen1);  // Defining the Hamming window
    float *fhamm = NULL;
    if (specfile)
    {
	fhamm = hamming(flen);
    }
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
	if (specfile)
	{
	    *spectrogram = (float*) MALLOC(nfeatfr * numframes * nbands * sizeof(float));
	}
	fprintf(stderr, "Parameters: (nframes=%d,  dim=%d)\n", numframes, *dim); 
    }

    int nframes = 0;
    struct fdlpenv_info* info = (struct fdlpenv_info*) MALLOC(nbands * sizeof(struct fdlpenv_info));
    for (int i = 0; i < nbands; i++ )
    {
	info[i].poles = p + i*(Np+1);
	info[i].Np = Np;
	info[i].fdlplen = fdlplen;
	info[i].ffdlplen = N;
	info[i].band = i;
	info[i].fnum = fnum;
	info[i].send = send1;
	info[i].fsend = send;
	info[i].flen = flen1;
	info[i].fflen = flen;
	info[i].fhop = fhop1;
	info[i].ffhop = fhop;
	info[i].mirrlen = mirr_len;
	info[i].fdlpwin = fdlpwin;
	info[i].fdlpolap = fdlpolap;
	info[i].nceps = *nceps;
	info[i].nframes = &nframes;
	info[i].energybands = energybands;
	info[i].feats = *feats;
	info[i].spectrogram = *spectrogram;
	info[i].hamm = hamm;
	info[i].fhamm = fhamm;

	band_threads[i].threadfunc = fdlpenv_pthread_wrapper;
	band_threads[i].thread_arg = (void*) &info[i];
    }

    run_threads(band_threads, nbands);

    if (do_spec)
    {
	if (axis == AXIS_LINEAR_MEL || axis == AXIS_LINEAR_BARK) {
	    audspec(&energybands, &nbands, nframes);
	}
	if (specgrm)
	{
	    //if (verbose) fprintf(stderr,"specgram flag =%d\n",specgrm);
	}
	else
	{
	    spec2cep4energy(energybands, nbands, nframes, *nceps, *feats, 0);
	}
    }

    FREE(info);

    //FREE(env_pad1);
    //FREE(env_pad2);
    //FREE(env_adpt);
    //FREE(env_log);
    FREE(p);
    FREE(energybands);
    FREE(hamm);		
    if (fhamm)
    {
	FREE(fhamm);
    }
}


int main(int argc, char **argv)
{ 
    parse_args(argc, argv);

#if HAVE_LIBPTHREAD == 1
    pthread_mutex_init(&fftw_mutex, NULL);
    pthread_mutex_init(&tpool_mutex, NULL);
    pthread_cond_init(&tpool_cond, NULL);
#endif

    LOOKUP_TABLE = (float*) MALLOC(((int) pow(2,nbits_log))*sizeof(float));
    fill_icsi_log_table(nbits_log,LOOKUP_TABLE); 

    int N;
    float *signal = readsignal_file(infile, &N);
    int Nsignal = N;

    fprintf(stderr, "Input file = %s; N = %d samples\n", infile, N);
    fprintf(stderr, "Gain Norm %d \n",do_gain_norm);
    fprintf(stderr, "Limit DCT range: %d\n", limit_range);
    fprintf(stderr, "Apply wiener filter: %d (alpha=%g)\n", do_wiener, wiener_alpha);

    fprintf(stderr, "Feature type: %s\n", (do_spec ? (specgrm ? "spectrogram" : "spectral") : "modulation"));
    fprintf(stderr, "FDPLP window length: %gs\n", fdplp_win_len_sec);

    int fwin = DEFAULT_SHORTTERM_WINLEN_MS * Fs;
    int fstep = DEFAULT_SHORTTERM_WINSHIFT_MS * Fs; 

    int fnum = floor(((float)N-fwin)/fstep)+1;
    //int fnum = floor(((float)N-fwin)/fstep); // stupid bug in feacalc...
    N = (fnum-1)*fstep + fwin;

    Nsignal = N;

    int fdlpwin = (int)(fdplp_win_len_sec * FDLPWIN_SEC2SHORTTERMMULT_FACTOR * fwin);
    int fdlpolap = DEFAULT_FDLPWIN_SHIFT_MS * Fs;  
    int nframes;
    //int add_samp;
    float *frames = fconstruct_frames(&signal, &N, fdlpwin, fdlpolap, &nframes);
    //add_samp = N - Nsignal;

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
	vad_labels = (int *)MALLOC(num_vad_labels * sizeof(int));
	for (int i = 0; i < num_vad_labels; i++) {
	    if (i < num_read_labels) {
		vad_labels[i] = (labels[i] == speechchar ? 1 : (labels[i] == nspeechchar ? 0 : 2));
		if (vad_labels[i] == 2) {
		    fprintf(stderr, "wrong char: <%c>\n", labels[i]);
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

    // DEBUG
    //FILE *fd = fopen("speech_signal.txt", "w");
    //for (int i = 0; i < N; i++) {
    //    fprintf(fd, "%d ", signal[i]);
    //}
    //fclose(fd);

    //fprintf(stderr, "Created %d frames of %d samples each, overlap %d\n", nframes, fdlpwin, fdlpolap);

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
    int nceps = num_cepstral_coeffs;
    int dim = 0;
    int nfeatfr_calculated = 0;

    float *feats = NULL;
    float *spectrogram = NULL;

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
	    local_size = Nsignal - f * (fdlpwin - fdlpolap);
	    if (local_size > Nsignal)
	    {
		local_size = Nsignal;
	    }
	    stop_before = 1;
	    xwin = signal + (Nsignal - local_size);
	}
	//fdither( xwin, local_size, 1 ); // TODO commented out
	sub_mean( xwin, local_size );

	int nfeatfr = (int)floor((local_size - fwin)/fstep)+1;

	if (feats == NULL)
	{
	    compute_fdlp_feats( xwin, local_size, Fs, &nceps, &feats, nfeatfr, nframes, &dim, &spectrogram );
	}
	else
	{
	    float *feat_mem = feats + nfeatfr_calculated * dim;
	    float *specgram_mem = spectrogram + (specfile == NULL ? 0 : nfeatfr_calculated * nbands);
	    compute_fdlp_feats( xwin, local_size, Fs, &nceps, &feat_mem, nfeatfr, nframes, &dim, &specgram_mem );
	}
	nfeatfr_calculated += nfeatfr;
	vad_label_start = nfeatfr_calculated;

	if (verbose || f == nframes - 1 || stop_before) fprintf(stderr, "%f s\n",toc());
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
    if (specfile)
    {
	fprintf(stderr, "Spectrogram output file = %s (%d frames, dimension %d)\n", outfile, fnum, nbands);
	writefeats_file(specfile, spectrogram, nbands, fnum);
    }

    // Free the heap
    cleanup_fftw_plans();
    if (vad_labels != NULL) {
	FREE(vad_labels);
    }
    FREE(signal);
    FREE(frames);
    FREE(feats);
    FREE(orig_wts);
    FREE(orig_indices);
    if (spectrogram != NULL)
	FREE(spectrogram);
    if (dctm)
	FREE(dctm);
    if (fft2decompm != NULL)
	FREE(fft2decompm);
    FREE(LOOKUP_TABLE);
    if (band_threads != NULL)
    {
	FREE(band_threads);
    }
#if HAVE_LIBPTHREAD == 1
    pthread_mutex_destroy(&fftw_mutex);
    pthread_mutex_destroy(&tpool_mutex);
    pthread_cond_destroy(&tpool_cond);
#else
    int mc = get_malloc_count();
    if (mc > 0)
	fprintf(stderr,"WARNING: %d malloc'd items not free'd\n", mc);
#endif
    fftw_forget_wisdom();

    return 0;
}
