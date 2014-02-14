#ifndef __TESTGEN
#define __TESTGEN

class Random;

void generateNIW(int dim, double *mu_0, double **wishart, double &rho, int &kappa);

void generateGauss(int dim, double *mu, double **cov, double *mu_0, double **wishart, double rho, int kappa, Random *random);

void generateGaussSample(int dim, double *eta, double *mu, double **cov, Random *random);

#endif
