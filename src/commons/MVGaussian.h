#pragma once

#include "random.h"

class MVGaussian
{
public:
	MVGaussian(uniform_random_t *uniform);
	~MVGaussian(void);

	void nextMVGaussian(double *mean, double **precision, double *res, const int &n);
	void nextMVGaussian(double *mean, double **precision, double **res, const int &n, const int &m);

	void nextMVGaussianWithCholesky(double *mean, double **precisionLowerTriangular, double *res, const int &n) ;

private:
	double nextGaussian();

private:
	// for Gaussian random variable
	int m_iSet;
	double m_dGset;
	uniform_random_t *uniform;
};
