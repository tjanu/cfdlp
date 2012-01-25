#ifndef CFDLP_H
#define CFDLP_H

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

#define DEFAULT_SHORTTERM_WINLEN_MS 0.025
#define DEFAULT_SHORTTERM_WINSHIFT_MS 0.010
#define DEFAULT_SHORTTERM_SHIFT_PERCENTAGE 0.4
#define DEFAULT_FDLPWIN_SHIFT_MS 0.020
#define FDLPWIN_SEC2SHORTTERMMULT_FACTOR 40
#define DEFAULT_LONGTERM_TRAP_FRAMECTX 10
#define DEFAULT_LONGTERM_WINLEN_MS 0.2

#define DEFAULT_SHORTTERM_NCEPS 13
#define DEFAULT_LONGTERM_NCEPS 14

# define pi 3.14159265

#endif/*CFDLP_H*/
