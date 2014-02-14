#include "polyagamma.h"
#include "defs.h"
#include <cmath>
#include <algorithm>
#include "Context.h"
#include "glog/logging.h"
using namespace std;

PolyaGamma::PolyaGamma( uniform_random_t *uniform )
{
	TRUNC = 0.64;
	cutoff = 1 / TRUNC;
	m_iSet = 0;

	this->uniform = uniform;
	this->npg = Context::get_instance().get_int("pgsamples");

	string modeStr = Context::get_instance().get_string("samplemode");

	if (modeStr=="precise")
	{
		mode = Precise;
	}
	else if (modeStr=="pg1")
	{
		mode = PG1;
	}
	else if (modeStr=="truncated")
	{
		mode = Truncated;
	}
	else if (modeStr=="gaussian")
	{
		mode = Gaussian;
	}
	else
	{
		LOG(FATAL) << "samplemode error: expected precise/pg1/truncated/gaussian, got " << modeStr << endl;
	}
}

PolyaGamma::~PolyaGamma(void)
{
}

// Sample from PG(n, Z) using Devroye-like method.
// n is a natural number and z is a positive real.
double PolyaGamma::nextPG(int n, double z)
{
	if (mode == PG1)
	{
		double dRes = 0;
		int nSmall = min(n, npg);
		//int nSmall = min(n, 16);
		//int nSmall = n;
		double rate = (double)n / nSmall;
		for ( int i=0; i<nSmall; i++ ) {
			dRes += nextPG1(z);
		}
	
		dRes *= rate;
		double zp = exp(z/2);
		double mean;
		
	 	if ( fabs(z) > 1e-7 )	
		{
			mean = (n*(zp-1/zp))/(z*(zp+1/zp)) * 0.5;
		}
		else 
		{
			// Using L'hospital's law
			mean = n * 0.25;
		}
	
		double error = (dRes - mean)/sqrt(rate);
	
		return mean + error;
	}
	else if (mode==Truncated)
	{
		// Must have this, or sometimes we'll get NaN when npg is small
		if (z > 100)
			z = 100;
		if (z < -100)
			z = -100;

		double sum = 0;
		double cc = z*z/(4*PI*PI);
		double coefficient	= 0;
		if (fabs(z) > 1e-3)
		{
			double zp		= exp(z/2);
			coefficient		= 1.0/(2*z) * (zp-1/zp)/(zp+1/zp);
		}
		else 
		{
			coefficient		= 0.25;
		}

		double correct_2	= 0;

		for (int k=1; k<=npg; ++k)
		{
			correct_2 += 1.0 / ( (k-0.5)*(k-0.5) + cc );
			sum += rgamma(n, 1) / ( (k-0.5)*(k-0.5) + cc );
		}

		sum = sum * coefficient / correct_2;

		if ( sum < 0 || isnan((double)sum) )
		{
			LOG(FATAL) << "negative sample " << sum << " n " << n << " z " << z << endl;
		}

		// DEBUG
		//double zp = exp(z/2);
		//double mean = (n*(zp-1/zp))/(z*(zp+1/zp)) * 0.5;
		//LOG(WARNING) << sum << " " << mean << endl;

		return sum;
	}
	else if (mode==Gaussian)
	{
		double sample;
		do
		{
			double exp0_5z  = exp(z/2);
			double iexp0_5z = 1.0 / exp0_5z;
			double expz	= exp0_5z * exp0_5z;
			double iexpz	= 1.0 / exp(z);
	
			double sinhz	= ( expz - iexpz ) / 2;
			double coshz	= ( expz + iexpz ) / 2;
			double sinh0_5z	= ( exp0_5z - iexp0_5z ) / 2;
			double cosh0_5z	= ( exp0_5z + iexp0_5z ) / 2;
			double tanh0_5z = sinh0_5z / cosh0_5z;
	
			double mean, ex2, var;
			if ( fabs(z) > 1e-3 )
			{
				mean = n / (2*z) * tanh0_5z;
	
				//ex2 = n * ( -(2+n)*z*z + n*z*z*coshz + 2*z*sinhz ) / (8*z*z*z*z * cosh0_5z * cosh0_5z);
				double sqrz = z*z;
				ex2 = n * ( (-2+(coshz-1)*n)*sqrz + 2*z*sinhz ) / (8*sqrz*sqrz * cosh0_5z * cosh0_5z);
			}
			else 
			{
				mean = (double) n / 4;
				ex2 = (double)n * (2 + 3*n) / 48;
			}
	
			var = ex2 - mean * mean;
			if (var<0)
			{
				LOG(WARNING) << "negative variance " << var << " n= " << n << " z= " << z << endl;
				var = 0;
			}
	
			sample = mean + sqrt(var) * rnorm();
		} while (sample < 0);

		return sample;
	}
	else
	{
		// Precise sampling
		double dRes = 0;
		for ( int i=0; i<n; i++ ) {
			dRes += nextPG1(z);
		}
		return dRes;
	}
}

double PolyaGamma::rgamma(double shape, double scale)
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
	   x = rgamma(shape+1, 1);
	   r = (*uniform)();
	   x = x*pow(r, 1/shape);
	   gamma1 = x / scale;
   }
   return gamma1;
}     	

// sample from PG(1, z)
double PolyaGamma::nextPG1(double zVal)
{
	double z = fabs(zVal) * 0.5;
	double fz = (PI * PI * 0.125 + z * z * 0.5);

	double X = 0;
	int numTrials = 0;
	while ( true ) 
	{
		numTrials ++;
		double dU = (*uniform)();
		if ( dU < texpon(z) ) {
			X = TRUNC + rexp(1) / fz;
		} else {
			X = rtigauss(z);
		}

		double S = a(0, X);
		double Y = (*uniform)() * S;
		int n = 0;
		while ( true ) {
			n ++;
			if ( n % 2 == 1 ) {
                S = S - a(n, X);
                if ( Y <= S ) break;
			} else {
               S = S + a(n,X);
               if ( Y>S ) break;
 			}
		};

       if ( Y<=S ) break;
 	};

	return 0.25 * X;
}

// rtigauss - sample from truncated Inv-Gauss(1/abs(Z), 1.0) 1_{(0, TRUNC)}.
double PolyaGamma::rtigauss(double Z)
{
	double R = TRUNC;

	Z = fabs(Z);
	double mu = 1 / Z;
	double X = R + 1;
	if ( mu > R ) {
		double alpha = 0;
		while ( (*uniform)() > alpha ) {
			double E1 = rexp(1);
			double E2 = rexp(1);
			while ( pow(E1,2.0) > 2*E2 / R) {
				E1 = rexp(1);
				E2 = rexp(1);
			}
			X = R / pow((1 + R*E1), 2.0);
			alpha = exp(-0.5 * Z * Z * X);
		}
	} else {
		while ( X > R ) {
			double lambda = 1;
			double Y = pow(rnorm(), 2.0);
			X = mu + 0.5*mu*mu / lambda * Y - 0.5 * mu / lambda * sqrt(4 * mu * lambda * Y + pow(mu*Y, 2.0));
			if ( (*uniform)() > mu / (mu + X) ) {
				X = pow(mu, 2.0) / X;
			}
		}
	}

    return X;
}

//// rigauss - sample from Inv-Gauss(mu, lambda).
//double PolyaGamma::rigauss(double mu, double lambda)
//{
//    double nu = rnorm(1);
//    double y  = pow(nu, 2.0);
//    double x  = mu + 0.5 * pow(mu, 2.0) * y / lambda -
//         0.5 * mu / lambda * sqrt(4 * mu * lambda * y + pow((mu*y), 2.0) );
//    if ( myrand48 > mu / (mu + x)) {
//        x = pow(mu, 2) / x;
//    }
//    return x;
//}

double PolyaGamma::texpon(double Z)
{
    double x = TRUNC;
    double fz = (PI*PI*0.125 + Z*Z*0.5);
    double b = sqrt(1.0 / x) * (x * Z - 1);
    double a = -1.0 * sqrt(1.0 / x) * (x * Z + 1);

    double x0 = log(fz) + fz * TRUNC;
    double xb = x0 - Z + pnorm(b, true);
    double xa = x0 + Z + pnorm(a, true);

    double qdivp = 4 / PI * ( exp(xb) + exp(xa) );

    return (1.0 / (1.0 + qdivp));
}

// the cumulative density function for standard normal
double PolyaGamma::pnorm(double x, bool bUseLog)
{
	const double c0 = 0.2316419;
	const double c1 = 1.330274429;
	const double c2 = 1.821255978;
	const double c3 = 1.781477937;
	const double c4 = 0.356563782;
	const double c5 = 0.319381530;
	const double c6 = 0.398942280401;
	const double negative = (x < 0 ? 1.0 : 0.0);
	const double xPos = (x < 0.0 ? -x : x);
	const double k = 1.0 / ( 1.0 + (c0 * xPos));
	const double y1 = (((((((c1*k-c2)*k)+c3)*k)-c4)*k)+c5)*k;
	const double y2 = 1.0 - (c6*exp(-0.5*xPos*xPos)*y1);

	if ( bUseLog ) {
		return log(((1.0-negative)*y2) + (negative*(1.0-y2)));
	} else {
		return ((1.0-negative)*y2) + (negative*(1.0-y2));
	}
}

// draw a sample from standard norm distribution.
double PolyaGamma::rnorm()
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
		return /*dMu + dSigma * */ v2 * dFac;
	} else {
		m_iSet = 0;
		return /*dMu + dSigma **/ m_dGset;
	}
}

// draw a sample from an exponential distribution with parameter lambda
double PolyaGamma::rexp(double lambda)
{
	double dval = 0;
	while ( dval >=1 || dval <= 0 ) {
		dval = (*uniform)();
	}
	return (-log(dval));// / lambda);
}


//double PolyaGamma::pigauss(x, mu, lambda)
//{
//    double Z = 1.0 / mu;
//    double b = sqrt(lambda / x) * (x * Z - 1);
//	double a = -1.0 * sqrt(lambda / x) * (x * Z + 1);
//    double y = exp(pnorm(b, log.p=TRUE)) + exp(2 * lambda * Z + pnorm(a, log.p=TRUE));
//
//	return y;
//}

#define square(x) ((x)*(x))

// Calculate coefficient n in density of PG(1.0, 0.0), i.e. J* from Devroye.
double PolyaGamma::a(int n, double x)
{
	double dRes = 0;

	if ( x>TRUNC )
		dRes = PI * (n+0.5) * exp( - square(n+0.5) * square(PI) * x / 2 );
	else
		dRes = pow((2/PI/x), 1.5) * PI * (n+0.5) * exp( -2* square(n+0.5) / x );

	return dRes;
}

// sample from normal distribution
