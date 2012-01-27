#ifndef HLPC_LS_H
#define HLPC_LS_H

#ifdef __cplusplus
extern "C" {
#endif
void hlpc_ls(double* y, int len, int order, int compr, float* poles);
#ifdef __cplusplus
};
#endif

#endif/*HLPC_LS_H*/
