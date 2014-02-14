#include "testgenerator.h"
#include "random.h"
#include "utils.h"

void generateNIW(int dim, double *mu_0, double **wishart, double &rho, int &kappa)
{
	int prior = 50; 
	rho = prior;
	kappa = prior;

	clear(mu_0, dim);
	clear2D(wishart, dim, dim);
	for (int i=0; i<dim; ++i)
		wishart[i][i] = prior;
}

void generateGauss(int dim, double *mu, double **cov, double *mu_0, double **wishart, double rho, int kappa, Random *random)
{
	random->NIWrnd(mu, cov, rho, kappa, mu_0, wishart, dim);
}

void generateGaussSample(int dim, double *eta, double *mu, double **cov, Random *random)
{
	double **prec = alloc2D<double>(dim, dim);
	inverse(cov, prec, dim);

	random->nextMVGaussian(mu, prec, eta, dim);

	free2D(prec, dim, dim);
}


