#ifndef __UTIL_H
#define __UTIL_H

#include <algorithm>
#include <memory.h>
#include <cstdio>
#include <stdio.h>
#include <cmath>
#include <float.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>
#include "types.h"
#include <google/protobuf/text_format.h>
#include <armadillo>
using namespace boost;

// Misc arithmetic
double logExpSum(double* eta, int n);
double logExpSum(double a, double b);
double logExpMinus(double a, double b);
double log_sum(double log_a, double log_b);
double log_sum(double* log_x, int n);
double lgamma(double x);
double vmax(double* x, int n);
int argmax(double* x, int n);

// Matrix and Vector manipulation
template<class T>
T* alloc(int n)
{
	T* ret = new T[n];
	memset(ret, 0, sizeof(T)*n);
	return ret;
}

template<class T>
T** alloc2D(int n, int m)
{
	T** ret = new T*[n];
	T* buff = new T[n*m];
	memset(buff, 0, sizeof(T)*n*m);
	for (int i=0; i<n; ++i)
		ret[i] = buff + i*m;
	return ret;
}

template<class T>
void copy2D(T** dest, T** src, int n, int m)
{
	for (int i=0; i<n; ++i)
		for (int j=0; j<m; ++j)
			dest[i][j] = src[i][j];
}

template<class T>
void copy2D(T** dest, int dn, int dm, T** src, int sn, int sm)
{
	int n = std::min(dn, sn);
	int m = std::min(dm, sm);
	for (int i=0; i<n; ++i)
		for (int j=0; j<m; ++j)
			dest[i][j] = src[i][j];
}

template<class T>
void clear(T* m, int n)
{
	for (int i=0; i<n; ++i)
		m[i] = 0;
}

template<class T>
void clear2D(T** m, int r, int c)
{
	for (int i=0; i<r; ++i)
		for (int j=0; j<c; ++j)
			m[i][j] = 0;
}

template<class T>
void free2D(T** m, int r, int c)
{
	delete[] m[0];
	delete[] m;
}

template<class T>
void swapRows(T **a, int n, int m, int *perm)
{
	T **temp = alloc2D<T>(n, m);
	copy2D<T>(temp, a, n, m);
	for (int i=0; i<n; ++i)
		for (int j=0; j<m; ++j)
		{
			a[i][j] = temp[perm[i]][j];
		}
		free2D<T>(temp, n, m);
}

template<class T>
void swapCols(T **a, int n, int m, int *perm)
{
	T **temp = alloc2D<T>(n, m);
	copy2D<T>(temp, a, n, m);
	for (int i=0; i<n; ++i)
		for (int j=0; j<m; ++j)
			a[i][j] = temp[i][perm[j]];
	free2D<T>(temp, n, m);
}

// Vector and Matrix arithmatic
double dotprod(double *a, double *b, const int&n);
void matrixprod(double *a, double **A, double *res, const int&n);
void matrixprod(double **A, double *a, double *res, const int&n);
// TODO These matrix functions are very dangerous to be confused, refactor this.
void addmatrix(double **A, double *a, double *b, const int &n, double factor);
void addmatrix(int **dest, int **src, const int &n, const int &m);
void addmatrix2(double **A, double *a, double *b, const int &n, double factor);
void addmatrix(double **src, double **dest, const int &n, double factor);
void addmatrix(double **src, double **dest, const int &n, const int &m, double factor);
void addvec(double *res, double *a, const int &n, double factor);
void addvec(double *res, double *a, const int &n);
void vmulti(double *res, const int &n, double factor);
void mmulti(double **res, const int &n, const int &m, double factor);
bool inverse(double **A, double **res, const int &n);
double inverse_det(double **A, double **res, const int &n);
bool choleskydec(double **A, double **res, const int &n, bool isupper);
void inverse_cholydec(double **A, double **res, double **lowerTriangle, const int &n);

// Misc
char* appendNum(const char *prefix, const char *extension, int num);
char* appendNum(const char *prefix, const char *extension);
void softmax(double *res, double *v, int n);

double perplexity(int D, int K, double **phi, double **theta, int **w, std::vector<int> length);
void readCorpus(const char *fileName, int **&w, std::vector<int> &length, int &D, int &W);

void create_generator(base_generator_type *&generator, uniform_real<> *&uni_dist, variate_generator<base_generator_type&, uniform_real<> > *&unif01);	

void print(const google::protobuf::Message &message);

int numOfSymmetryMat(int dim);
void symmMatToArray(arma::mat m, double *buff); 
arma::mat arrayToSymmMat(double *buff, int dim);

arma::mat MPI_Reduce_Symmetry_Matrix(arma::mat &source, int rank, int size);
void MPI_Broadcast_Symmetry_Matrix(arma::mat &work, int rank, int size);

#endif
