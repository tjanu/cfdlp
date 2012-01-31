#define NDEBUG
#include "hlpc_ls.h"
#include "invert_matrix.hpp"
//#include "invert_matrix_gj.hpp"
#include "stopwatch.hpp"
#include <cmath>
#include <limits>
#include <fstream>

#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#if HLPC_DO_TIMING == 1
#define TIME_THIS(x, y) cxxutil::stopwatch x(y)
#else
#define TIME_THIS(x, y)
#endif

typedef ublas::matrix<double> dmatrix;
typedef ublas::vector<double> dvector;

void toeplitz(const dvector& c, const dvector& r, dmatrix& result) {
    TIME_THIS(watch, "toeplitz");
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
    TIME_THIS(watch, "geninv");
    int m = mat.size1();
    int n = mat.size2();
    dmatrix A;
    bool transpose = false;
    {
	TIME_THIS(getA, "get A");
	if (m < n) {
	    transpose = true;
	    A = prod(mat, ublas::trans(mat));
	    n = m;
	} else {
	    A = prod(ublas::trans(mat), mat);
	}
    }

    typedef ublas::banded_matrix<double> diag_matrix;
    diag_matrix dA(A.size1(), A.size2(), 0, 0);
    ublas::matrix_vector_slice<dmatrix> A_diag(A, ublas::slice(0, 1, A.size1()), ublas::slice(0, 1, A.size2()));
    { TIME_THIS(getdA, "get dA");
	for (int i = 0; i < dA.size1(); i++) {
	    if (i < dA.size2()) {
		dA(i, i) = A_diag(i);
	    }
	}
    }
    double minimum = std::numeric_limits<double>::max();
    {
	TIME_THIS(getMin,"get minimum");
	for (int i = 0; i < dA.size1(); i++) {
	    if (i < dA.size2()) {
		if (dA(i, i) > 0.) {
		    if (minimum > dA(i, i)) {
			minimum = dA(i, i);
		    }
		}
	    }
	}
    }
    double tol = minimum * 1e-9;
    dmatrix L(A.size1(), A.size2());
    {
	TIME_THIS(getL, "get L");
	for (int i = 0; i < L.size1(); i++) {
	    for (int j = 0; j < L.size2(); j++) {
		L(i, j) = 0.;
	    }
	}
    }
    int r = 0;

    {
	TIME_THIS(chol, "cholesky_decomposition");
	for (int k = 0; k < n; k++) {
	    r++;
	    ublas::vector<double> nullvec(n-k);
	    if (r > 1) {
		ublas::project(L, ublas::range(k, n), ublas::range(r-1, r))
		    = ublas::project(A, ublas::range(k, n), ublas::range(k, k+1))
		    - ublas::prod(
			    ublas::project(L, ublas::range(k, n), ublas::range(0, r)),
			    ublas::trans(ublas::project(L, ublas::range(k, k+1), ublas::range(0, r)))
			    );
		dmatrix projection(ublas::prod(ublas::project(L, ublas::range(k, n), ublas::range(0, 3)), 
			    ublas::trans(ublas::project(L, ublas::range(k, k+1), ublas::range(0, 3)))));
	    } else {
		ublas::project(L, ublas::range(k, n), ublas::range(r-1, r))
		    = ublas::project(A, ublas::range(k, n), ublas::range(k, k+1));
		dmatrix projection(ublas::project(A, ublas::range(k, n), ublas::range(k, k+1)));
	    }
	    if (L(k, r-1) > tol) {
		L(k, r-1) = std::sqrt(L(k, r-1));
		if (k < n-1) {
		    ublas::project(L, ublas::range(k+1, n), ublas::range(r-1, r))
			= ublas::project(L, ublas::range(k+1, n), ublas::range(r-1, r)) / L(k, r-1);
		}
	    } else {
		r--;
	    }
	}
    }
    { TIME_THIS(projecting, "projecting L");
	L = project(L, ublas::range(0, L.size1()), ublas::range(0, r));
    }
    dmatrix p(ublas::prod(ublas::trans(L), L));
    dmatrix M(p.size1(), p.size2());
    {
	TIME_THIS(inverting, "inverting matrix");
	invert_matrix(p, M);
    }
//    {
//	TIME_THIS(inverting, "inverting matrix via gauss-jordan");
//	bool b = invert_matrix_gj(p, M);
//	if (!b) {
//	    std::cerr << "matrix inversion via gj failed." << std::endl;
//	}
//    }

    {
	TIME_THIS(matmult, "final multiplications");
	if (transpose) {
	    inverse = ublas::prod(ublas::trans(mat), L);
	    inverse = ublas::prod(inverse, M);
	    inverse = ublas::prod(inverse, M);
	    inverse = ublas::prod(inverse, ublas::trans(L));
	} else {
	    inverse = ublas::prod(L, M);
	    inverse = ublas::prod(inverse, M);
	    inverse = ublas::prod(inverse, ublas::trans(L));
	    inverse = ublas::prod(inverse, ublas::trans(mat));
	}
    }
}

void hlpc_ls(double* y, int len, int order, int compr, float* poles)
{
    TIME_THIS(hlpc_watch, "hlpc_ls");
    int col_length = len - order;
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
    TIME_THIS(watch, "hlpc_finalize");
    dvector tmp_a = prod(inv, a_multiplier);
    double err = ublas::norm_2(a_multiplier - prod(R, tmp_a));
    if (isnan(err)) {
	std::cerr << "err is nan!" << std::endl;
	std::ofstream out("broken_inv.txt", std::ios::out);
	out << inv << std::endl;
	out.close();
	out.open("broken_R.txt", std::ios::out);
	out << R << std::endl;
	out.close();
	out.open("broken_column.txt", std::ios::out);
	out << column << std::endl;
	out.close();
	out.open("broken_row.txt", std::ios::out);
	out << row << std::endl;
	out.close();
	out.open("broken_y.txt", std::ios::out);
	for (int k = 0; k < len; k++) {
	    out << y[k] << std::endl;
	}
	out.close();
	std::cerr << "y_len=" << len << ", order=" << order << std::endl;
	std::exit(1);
    }
    dvector a(1 + tmp_a.size());
    a(0) = 1.;
    for (int i = 1; i < a.size(); i++) {
	a(i) = -tmp_a(i-1);
    }
    a = a / sqrt(err);

    // copy over to output
    for (int i = 0; i < a.size(); i++) {
	poles[i] = a(i);
	if (isnan(poles[i])) {
	    std::cerr << "poles[" << i << "] is nan!" << std::endl;
	    std::ofstream out("broken_inv.txt", std::ios::out);
	    out << inv << std::endl;
	    out.close();
	    out.open("broken_R.txt", std::ios::out);
	    out << R << std::endl;
	    out.close();
	    out.open("broken_column.txt", std::ios::out);
	    out << column << std::endl;
	    out.close();
	    out.open("broken_row.txt", std::ios::out);
	    out << row << std::endl;
	    out.close();
	    out.open("broken_y.txt", std::ios::out);
	    for (int k = 0; k < len; k++) {
		out << y[k] << std::endl;
	    }
	    out.close();
	    out.open("broken_a.txt", std::ios::out);
	    out << a << std::endl;
	    out.close();
	    out.open("broken_tmp_a.txt", std::ios::out);
	    out << tmp_a << std::endl;
	    out.close();
	    std::cerr << "y_len=" << len << ", order=" << order << ", err=" << err << std::endl;
	    std::exit(1);
	}
    }
}
