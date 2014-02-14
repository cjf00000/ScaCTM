#ifndef __POLYAGAMMA
#define __POLYAGAMMA
#include "random.h"

// Sample from PG(n, Z) using Devroye-like method.
class PolyaGamma
{
public:
	PolyaGamma(uniform_random_t *uniform);
	~PolyaGamma(void);

	double nextPG(int n, double z);
	double nextPG1(double z);

	double texpon(double Z);
	double rtigauss(double Z);
	double rigauss(double mu, double lambda);
	double rgamma(double shape, double scale);
	double a(int n, double x);

	double rnorm();
	double rexp(double lambda);
	double pnorm(double x, bool bUseLog);

private:
	enum SampleMode { Precise, PG1, Truncated, Gaussian };

	double TRUNC;
	double cutoff;

	// for Gaussian random variable
	double m_dGset;
	int m_iSet;
	int npg;
	SampleMode mode;

	uniform_random_t *uniform;
};

#endif
