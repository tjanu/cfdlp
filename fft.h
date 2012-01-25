#ifndef FFT_H
#define FFT_H

struct fft_info_;

typedef struct fft_info_* fft_info;

fft_info fft_info_r2r_new(void);
fft_info fft_info_r2c_new(void);
fft_info fft_info_c2r_new(void);

void fft_info_prepare(fft_info fft, int size, double* input, double* output

#endif/*FFT_H*/
