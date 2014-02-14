#include "MVGaussian2.h"
#include "utils.h"
#include "defs.h"
#include "utils.h"
#include <ctime>
#include <iostream>
#include <pthread.h>
using namespace std;
using namespace arma;

#define tic timer.tic()
#define toc cerr << timer.toc() << endl;

class MatGaussianFiller
{
public:
	MatGaussianFiller()
	{
		m_iSet = 0;
		create_generator(generator, uni_dist, uniform);

		int tmp = rand()&1023;
		for (int i=0; i<tmp; ++i)
			(*uniform)();
	}

	~MatGaussianFiller()
	{
		delete generator;
		delete uni_dist;
		delete uniform;
	}

	void operator() (arma::mat *res, int beginCol, int endCol)
	{
		int nrows = res->n_rows;
		double *mem = res->memptr();

		for (int i=beginCol; i<endCol; ++i)
			for (int j=0; j<nrows; ++j)
				mem[i*nrows + j] = nextGaussian();
	//			(*res)(j, i) = nextGaussian();
	}

private:
	double nextGaussian()
	{
		if ( m_iSet == 0 ) {
			double dRsq = 0;
			double v1, v2;
			do {
				v1 = 2.0 * (*uniform)() - 1.0;
				v2 = 2.0 * (*uniform)() - 1.0;
				dRsq = v1 * v1 + v2 * v2;
			} while (dRsq > 1.0 || dRsq < 1e-300);
	
			double dFac = sqrt(-2.0 * log(dRsq) / dRsq);
			m_dGset = v1 * dFac;
			m_iSet = 1;
			return v2 * dFac;
		} else {
			m_iSet = 0;
			return m_dGset;
		}
	}

	int m_iSet;
	double m_dGset;

	base_generator_type *generator;
	uniform_real<> *uni_dist;
	uniform_random_t *uniform;
};


MVGaussian2::MVGaussian2()
{
	//For load balancing
	//TODO Ugly, use mutex to assign tasks dynamically
	nthreads = 240;
	//nthreads = 1;
	filler = new MatGaussianFiller[nthreads];
}

MVGaussian2::~MVGaussian2(void)
{
	delete[] filler;
}

struct FillTask
{
	int begin;
	int end;
	arma::mat *y;
	MatGaussianFiller *filler;
};

void* fillthread(void *args)
{
	FillTask* task = (FillTask*)args;

	(*(task->filler))(task->y, task->begin, task->end);

	pthread_exit(NULL);
}

void MVGaussian2::parallelFill(arma::mat *y)
{
	int ncols = y->n_cols;

	int chuckSize = (ncols-1) / nthreads + 1;

	int lastEnd = 0;

	pthread_t *threads = new pthread_t[nthreads];
	FillTask *tasks = new FillTask[nthreads];
	for (int k=0; k<nthreads; ++k)
	{
		int myBegin = lastEnd;
		int myEnd = min(ncols, myBegin + chuckSize);
		lastEnd = myEnd;

		tasks[k].begin = myBegin;
		tasks[k].end = myEnd;
		tasks[k].y = y;
		tasks[k].filler = &filler[k];

		pthread_create(&threads[k], NULL, fillthread, &tasks[k]);
	}

	for (int k=0; k<nthreads; ++k)
	{
		pthread_join(threads[k], NULL);
	}

	delete[] threads;
	delete[] tasks;
}

arma::mat MVGaussian2::nextMVGaussian(arma::mat &mu,
					arma::mat &precision,
					const int n)
{
	int dim = mu.n_rows;

	mat u = chol(precision);

	mat y = mat(dim, n);
	parallelFill(&y);

	mat x = solve(u, y);

	for (int k=0; k<n; ++k)
		for (int d=0; d<dim; ++d)
			x(d, k) += mu(d, 0);

	return x;
}
