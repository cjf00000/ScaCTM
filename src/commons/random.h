#ifndef __RNG_H
#define __RNG_H

#include "types.h"
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>
#include <armadillo>
using namespace boost;

class MVGaussian;
class MVGaussian2;
class PolyaGamma;

class Random
{
public:	
	Random(uniform_random_t *uniform);
	~Random();

	void randvector(double *a, const int n, double factor);
	void randpdmatrix(double **a, const int n, const int dof, double factor);

	double gammarnd(double shape, double scale);
	void dirichletrnd(double *a, const int N, double *theta);
	double nextPG(int n, double z);
	double rnorm();
//	double rand();
//	unsigned long long longRand();

	int multRnd(double *theta, int D);
	void nextMVGaussian(double *mean, double **precision, double *res, const int &n);
	void nextMVGaussian(double *mean, double **precision, double **res, const int &n, const int &m);
	arma::mat nextMVGaussian(arma::mat &mu,
			    arma::mat &precision,
			    const int n);

	void rinvertwishart (double **wishart, double **LAMDA, int n, int m);
	void rinvertwishart (double **wishart, double **LAMDA, int n, int m, int rank, int size);
	arma::mat rinvertwishart(arma::mat &LAMBDA, int m, int rank, int size);

	// Generate normal-inverse-wishart
	// E[cov] = wishart / (kappa-N-1)
	void NIWrnd(double *mu, double **cov,
		double rho, int kappa, double *mu_0, double **wishart,
		int N);
	void NIWrnd(double *mu, double **cov,
		double rho, int kappa, double *mu_0, double **wishart,
		int N, int rank, int size);

	// mu, cov are empty matrix of dim dimensions
	void NIWrnd(arma::mat &mu, arma::mat &cov,
		double rho, int kappa, 
		arma::mat &mu_0, arma::mat &wishart,
		int rank, int size);

private:
	void mpi_iw_xtx(double **temp2, double **LAMDA, int dim, int m, int rank, int size);
	arma::mat mpi_iw_xtx(arma::mat &LAMBDA, int m, int rank, int size);
	MVGaussian2 *fast_gaussian;
	MVGaussian *gaussian;
	PolyaGamma *pg;
	uniform_random_t *uniform;
};

#endif
