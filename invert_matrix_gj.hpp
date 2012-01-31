#ifndef INVERT_MATRIX_GJ_HPP
#define INVERT_MATRIX_GJ_HPP

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

#define abs(x) ((x) < 0 ? -x : x)

namespace ublas = boost::numeric::ublas;

template<class T>
bool invert_matrix_gj(const ublas::matrix<T>& input, ublas::matrix<T>& inverse) {
    typedef ublas::matrix<T> tmatrix;
    typedef ublas::vector<T> tvector;
    typedef ublas::identity_matrix<T> imatrix;

    T eps = 1e-10;

    int h = input.size1();
    int w = input.size2();

    // get a working copy of the input and augment it by the identity matrix
    tmatrix A(h, w + h);
    imatrix I(h, h);
    for (int i = 0; i < A.size1(); i++) {
	for (int j = 0; j < A.size2(); j++) {
	    if (j < input.size2()) {
		A(i, j) = input(i, j);
	    } else {
		A(i, j) = I(i, j - input.size2());
	    }
	}
    }

    // do gauss-jordan
    for (int y = 0; y < h; y++) {
	// find max pivot
	int maxrow = y;
	for (int y2 = y+1; y2 < h; y2++) {
	    if (abs(A(y2, y)) > abs(A(maxrow,y))) {
		maxrow = y2;
	    }
	}
	// swap row y with maxrow
	ublas::matrix_row<tmatrix> current(A, y);
	ublas::matrix_row<tmatrix> toswap (A, maxrow);
	current.swap(toswap);
	// check for singularity
	if (abs(A(y, y)) <= eps) {
	    return false;
	}
	// eliminate row y
	for (int y2 = y+1; y2 < h; y2++) {
	    T c = A(y2, y) / A(y, y);
	    for (int x = y; x < w+h; x++) {
		A(y2, x) -= A(y, x) * c;
	    }
	}
    }
    // backsubstitution
    for (int y = h - 1; y >= 0; y--) {
	T c = A(y, y);
	for (int y2 = 0; y2 < y; y2++) {
	    for (int x = w+h - 1; x >= 0; x--) {
		A(y2, x) -= A(y, x) * A(y2, y) / c;
	    }
	}
	A(y, y) /= c;
	for (int x = h; x < w+h; x++) {
	    A(y, x) /= c;
	}
    }
    // get out the inverted part
    ublas::matrix_range<tmatrix> orig_range(A, ublas::range(0, h), ublas::range(h, w+h));
    inverse = orig_range;
    return true;
}

#endif/*INVERT_MATRIX_GJ_HPP*/
