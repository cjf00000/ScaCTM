#pragma once

#include "random.h"
#include <armadillo>

class MatGaussianFiller;

class MVGaussian2
{
public:
	MVGaussian2();
	~MVGaussian2(void);

	// mu		:	dim x 1
	// precision	:	dim x dim
	// return	:	dim x n
	arma::mat nextMVGaussian(arma::mat &mu,
			    arma::mat &precision,
			    const int n);

private:
	void parallelFill(arma::mat *y);

	int nthreads;
	MatGaussianFiller *filler;
};
