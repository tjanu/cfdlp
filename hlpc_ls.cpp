#include "hlpc_ls.h"
#include "invert_matrix.hpp"
#include <cmath>
#include <limits>

typedef ublas::matrix<double> dmatrix;
typedef ublas::vector<double> dvector;

void toeplitz(const dvector& c, const dvector& r, dmatrix& result) {
    result.resize(c.size(), r.size(), false);

    for (int i = 0; i < result.size1(); i++) {
	for (int j = 0; j < result.size2(); j++) {
	    if (j <= i) {
		result(i, j) = c(i-j);
	    } else {
		result(i, j) = r(j-i);
	    }
	}
    }
}

void geninv(const dmatrix& mat, dmatrix& inverse) {
    int m = mat.size1();
    int n = mat.size2();
    dmatrix A;
    bool transpose = false;
    if (m < n) {
	transpose = true;
	A = mat * ublas::trans(mat);
	n = m;
    } else {
	A = ublas::trans(mat) * mat;
    }

    typedef ublas::banded_matrix<double> diag_matrix;
    diag_matrix dA(A.size1(), A.size2(), 0, 0);
    dA = A;
    double minimum = std::numeric_limits<double>::max();
    for (diag_matrix::iterator1 i = dA.begin(); i != dA.end(); i++) {
	if ((*i) > 0.) {

	}
    }
    double tol = 0.;
    dmatrix L(A.size1(), A.size2());
    int r = 0;
#if 0
function Y = geninv(G)
% Returns the Moore-Penrose inverse of the argument
% Transpose if m < n
[m,n]=size(G); transpose=false;
if m<n 
    transpose=true;
    A=G*G';
    n=m;
else
    A=G'*G;
end
    % Full rank Cholesky factorization of A
dA=diag(A); tol= min(dA(dA>0))*1e-9;
L=zeros(size(A));
r=0;
for k=1:n
    r=r+1;
    L(k:n,r)=A(k:n,k)-L(k:n,1:(r-1))*L(k,1:(r-1))';
    % Note: for r=1, the substracted vector is zero
    if L(k,r)>tol
        L(k,r)=sqrt(L(k,r));
        if k<n 
            L((k+1):n,r)=L((k+1):n,r)/L(k,r);
        end 
    else
        r=r-1;
    end 
end
L=L(:,1:r);
    % Computation of the generalized inverse of G
M=inv(L'*L);
if transpose
    Y=G'*L*M*M*L';
else
    Y=L*M*M*L'*G';
end
#endif
}

void hlpc_ls(double* y, int len, int order, int compr, float* poles)
{
    int col_length = len - order - 1;
    int row_length = order;

    dvector column(col_length);
    dvector row(row_length);
    dvector a_multiplier(col_length);

    for (int i = 0; i < col_length; i++) {
	column(i) = y[order - 1 + i];
	a_multiplier(i) = y[order + i];
    }
    for (int i = 0; i < row_length; i++) {
	row(i) = y[order - 1 - i];
    }

    dmatrix R;
    toeplitz(column, row, R);
    dmatrix inv;
    geninv(R, inv);
    //dvector tmp_a = inv * a_multiplier;
    //double err = ublas::norm_2(a_multiplier - R * tmp_a);
    //dvector a(1 + tmp_a.size());
    //a(0) = 1.;
    //for (int i = 1; i < a.size(); i++) {
    //    a(i) = -tmp_a(i-1);
    //}
    //a = a / sqrt(err);
}
