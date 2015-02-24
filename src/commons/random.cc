#include <mpi.h>
#include "random.h"
#include "MVGaussian.h"
#include "MVGaussian2.h"
#include "polyagamma.h"
#include "utils.h"
#include "matrixIO.h"
#include "glog/logging.h"
using namespace arma;

Random::Random(uniform_random_t *uniform)
{
	gaussian = new MVGaussian(uniform);
	pg = new PolyaGamma(uniform);
	this->uniform = uniform;

	fast_gaussian = new MVGaussian2();
}

Random::~Random()
{
	delete gaussian;
	delete pg;
	delete fast_gaussian;
}

void Random::randvector(double *a, const int n, double factor)
{
	for (int i=0; i<n; ++i)
		a[i] = (*uniform)() * factor;
}

void Random::randpdmatrix(double **a, const int n, const int dof, double factor)
{
	double **base = alloc2D<double>(n, dof);
	for (int i=0; i<n; ++i)
		for (int j=0; j<dof; ++j)
			base[i][j] = ((*uniform)() - 0.5) * factor;

	clear2D<double>(a, n, n);
	for (int i=0; i<n; ++i)
		for (int j=0; j<n; ++j)
			for (int k=0; k<dof; ++k)
				a[i][j] += base[i][k] * base[j][k];

	free2D<double>(base, n, dof);
}

double Random::gammarnd(double shape, double scale)
{
   double  b, h, r, g, f, x, r2, d, gamma1=10;

   b = shape - 1.0;
   
   if (b >=0) {
	   h = sqrt (3 * shape - 0.75);
       do {
          do {
             do {
		r = (*uniform)();
                g = r - pow(r, 2);
             } while (g <= 0.0);
             f = (r - 0.5) * h /sqrt(g);
   			 x = b + f;
           } while (x <= 0.0);
   			r2 = (*uniform)();
   			d = 64 * g * (pow(r2*g, 2));
   			if (d <= 0) {
               gamma1 = x;
               break;
            }
   			if (d*x < (x - 2*f*f)) {
                gamma1 = x;
                 break;
            }
       } while (log(d) > 2*(b*log(x/b) - f));
       gamma1 = x;
       gamma1 = (1 / scale) * gamma1;
   }
   else if (b < 0) {
	   x = gammarnd (shape+1, 1);
	   r = (*uniform)();
	   x = x*pow(r, 1/shape);
	   gamma1 = x / scale;
   }
   return gamma1;
}     	

void Random::dirichletrnd(double *a, const int N, double *theta)
{
	double sum = 0;
	for (int i=0; i<N; ++i)
		sum += a[i] = gammarnd(theta[i], 1);
	for (int i=0; i<N; ++i)
		a[i] /= sum;
}

double Random::nextPG(int n, double z)
{
	return pg->nextPG(n, z);
}

double Random::rnorm()
{
	return pg->rnorm();
}

int Random::multRnd(double *theta, int D)
{
	double r = (*uniform)();
	for (int i=0; i<D; ++i)
		if ( (r-=theta[i]) <= 0 )
			return i;
	return D-1;
}

void Random::nextMVGaussian(double *mean, double **precision, double *res, const int &n)
{
	return gaussian->nextMVGaussian(mean, precision, res, n);
}

void Random::nextMVGaussian(double *mean, double **precision, double **res, const int &n, const int &m)
{
	return gaussian->nextMVGaussian(mean, precision, res, n, m);
}

arma::mat Random::nextMVGaussian(arma::mat &mu,
			    arma::mat &precision,
			    const int n)
{
	return fast_gaussian->nextMVGaussian(mu, precision, n);
}

// compute temp2 = X^t X, where X is a m * dim matrix, each row of X is a sample from N(X_r | 0, LAMDA^-1)
// 1. Broadcast LAMDA
// 2. Decide num_local_samples
// 3. Draw num_local_samples MVGaussian
// 4, Compute X^T X locally
// 5. Reduce
// @note Caller should set temp2 to zero for us
void Random::mpi_iw_xtx(double **temp2, double **LAMDA, int dim, int m, int rank, int size)
{
	int must_have = m / size;
	int remainder = m % size;

	int num_local_samples = must_have + (rank < remainder);

	double *mean = alloc<double>(dim);
	double **X = alloc2D<double>(num_local_samples, dim);
	double **local_temp2 = alloc2D<double>(dim, dim);
	nextMVGaussian(mean, LAMDA, X, dim, num_local_samples);

	for (int k=0; k<num_local_samples; ++k)
		for (int i=0; i<dim; ++i)
			for (int j=0; j<dim; ++j)
				local_temp2[i][j] += X[k][i] * X[k][j];

	MPI_Reduce(*local_temp2, temp2 ? *temp2 : NULL, dim*dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	delete[] mean;
	free2D(X, num_local_samples, dim);
	free2D(local_temp2, dim, dim);
}

arma::mat Random::mpi_iw_xtx(arma::mat &LAMBDA, int m, int rank, int size)
{
	int dim = LAMBDA.n_rows;

	int must_have = m / size;
	int remainder = m % size;

	int num_local_samples = must_have + (rank < remainder);

	mat mean = zeros<mat>(dim, 1);

	mat X = nextMVGaussian(mean, LAMBDA, num_local_samples);
	mat local_temp2 = X * X.t();

	return MPI_Reduce_Symmetry_Matrix(local_temp2, rank, size);
}

void Random::rinvertwishart(double **wishart, double **LAMDA, int n, int m, int rank, int size)
{
	int dim = n;
	double **temp2 = NULL;
	
	if (rank==0)
	{
		temp2 = alloc2D<double>(dim, dim);
	}

	mpi_iw_xtx(temp2, LAMDA, dim, m, rank, size);

	if (rank==0)
	{
		inverse(temp2, wishart, dim);
	}
	
	if (rank==0)
	{
		free2D<double>(temp2, dim, dim);
	}
}

mat Random::rinvertwishart(mat &LAMBDA, int m, int rank, int size)
{
	int dim = LAMBDA.n_rows;

	mat wishart = mpi_iw_xtx(LAMBDA, m, rank, size);

	if (rank==0)
		return wishart.i();
	else
		return mat(dim, dim);
}

void Random::rinvertwishart(double **wishart, double **LAMDA, int n, int m)
{
	int dim = n;
	double *mean = alloc<double>(dim);
	double **X = alloc2D<double>(m, dim);
	double **temp2 = alloc2D<double>(dim, dim);

	for (int i = 0; i < dim; i++) mean[i] = 0.0;
	nextMVGaussian(mean, LAMDA, X, dim, m);
	
	for (int i=0; i<dim; ++i)
		for (int j=0; j<dim; ++j)
			for (int k=0; k<m; ++k)
				temp2[i][j] += X[k][i] * X[k][j];

	inverse(temp2, wishart, dim);
	
	delete[] mean;
	free2D<double>(X, m, dim);
	free2D<double>(temp2, dim, dim);
}

void Random::NIWrnd(double *mu, double **cov,
	double rho, int kappa, double *mu_0, double **wishart,
	int N)
{
	rinvertwishart(cov, wishart, N, kappa);

	double **p = alloc2D<double>(N, N);
	double **q = alloc2D<double>(N, N);

	copy2D<double>(p, cov, N, N);
	mmulti(p, N, N, 1.0/rho);	// p = cov / rho
	
	inverse(p, q, N);		// q = p^-1

	nextMVGaussian(mu_0, q, mu, N);

	free2D<double>(p, N, N);
	free2D<double>(q, N, N);
}

// if rank=0, wishart should be the parameter, otherwise it should be an arbitary matrix
void Random::NIWrnd(double *mu, double **cov,
	double rho, int kappa, double *mu_0, double **wishart,
	int N, int rank, int size)
{
	MPI_Bcast(*wishart, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&kappa, 1, MPI_INT, 0, MPI_COMM_WORLD);

	rinvertwishart(cov, wishart, N, kappa, rank, size);

	if (rank==0)
	{
		double **p = alloc2D<double>(N, N);
		double **q = alloc2D<double>(N, N);
	
		copy2D<double>(p, cov, N, N);
		mmulti(p, N, N, 1.0/rho);	// p = cov / rho
		
		inverse(p, q, N);		// q = p^-1
	
		nextMVGaussian(mu_0, q, mu, N);
	
		free2D<double>(p, N, N);
		free2D<double>(q, N, N);
	}
}

void Random::NIWrnd(arma::mat &mu, arma::mat &cov,
		double rho, int kappa, 
		arma::mat &mu_0, arma::mat &wishart,
		int rank, int size)
{
	int dim = mu.n_rows;

	wall_clock start;
	start.tic();
	MPI_Broadcast_Symmetry_Matrix(wishart, rank, size);
	MPI_Bcast(&kappa, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// No need to broadcast mu_0, since we only need it on rank 0

	start.tic();
	cov = rinvertwishart(wishart, kappa, rank, size);

	if (rank==0)
	{
		start.tic();
		mat prec = (cov/rho).i();

		mu = nextMVGaussian(mu_0, prec, 1);
	}
}
