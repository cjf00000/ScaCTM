// (C) Copyright 2009, Jun Zhu (junzhu [at] cs [dot] cmu [dot] edu)

// This file is part of MedLDA.

// MedLDA is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// MedLDA is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include <mpi.h>
#include "utils.h"
#include "spdinverse.h"
#include <time.h>
#include "defs.h"
#include <cassert>
#include "matrixIO.h"
#include <google/protobuf/text_format.h>
#include <string>
#include <iostream>
#include "glog/logging.h"
using namespace std;
using namespace arma;

double get_runtime(void)
{
  /* returns the current processor time in hundredth of a second */
  clock_t start;
  start = clock();
  return ((double)start/((double)(CLOCKS_PER_SEC)/100.0));
}

/*
* given log(a) and log(b), return log(a + b)
*/
double log_sum(double log_a, double log_b)
{
	if (log_a < log_b)
		return log_b+log(1 + exp(log_a-log_b));
	else 
		return log_a+log(1 + exp(log_b-log_a));
}

double log_sum(double* log_x, int n)
{
	double mx = vmax(log_x, n);
	int k = n;
	double ret = 0;
	double* ptr = log_x;
	while(k--)
		ret += exp(*ptr++ - mx);	
	ret = mx + log(ret);
	return ret;
}

double lgamma(double x)
{
	double x0,x2,xp,gl,gl0;
	int n,k;
	static double a[] = {
		8.333333333333333e-02,
		-2.777777777777778e-03,
		7.936507936507937e-04,
		-5.952380952380952e-04,
		8.417508417508418e-04,
		-1.917526917526918e-03,
		6.410256410256410e-03,
		-2.955065359477124e-02,
		1.796443723688307e-01,
		-1.39243221690590
	};

	x0 = x;
	if (x <= 0.0) return 1e308;
	else if ((x == 1.0) || (x == 2.0)) return 0.0;
	else if (x <= 7.0) {
		n = (int)(7-x);
		x0 = x+n;
	}
	x2 = 1.0/(x0*x0);
	xp = 2.0*PI;
	gl0 = a[9];
	for (k=8;k>=0;k--) {
		gl0 = gl0*x2 + a[k];
	}
	gl = gl0/x0+0.5*log(xp)+(x0-0.5)*log(x0)-x0;
	if (x <= 7.0) {
		for (k=1;k<=n;k++) {
			gl -= log(x0-1.0);
			x0 -= 1.0;
		}
	}
	return gl;
}

/*
* argmax
*/
int argmax(double* x, int n)
{
	int i;
	double max = x[0];
	int argmax = 0;
	for (i = 1; i < n; i++)
	{
		if (x[i] > max)
		{
			max = x[i];
			argmax = i;
		}
	}
	return(argmax);
}

double vmax(double* x, int n)
{
	double mx = *x;
	int k = n - 1;
	double* ptr = x + 1;
	while(k--)
	{
		if (*ptr > mx) mx = *ptr++;
		else ptr++;
	}
	return mx;
}

double dotprod(double *a, double *b, const int&n)
{
	int k = n;
	double res = 0;
	double* ptr_a = a;
	double* ptr_b = b;
	while(k--)
		res += (*ptr_a++) * (*ptr_b++);
	return res;
}

/* a vector times a (n x n) square matrix  */
void matrixprod(double *a, double **A, double *res, const int &n)
{
	for ( int i=0; i<n; i++ ) {
		res[i] = 0;
		for ( int j=0; j<n; j++ ) {
			res[i] += a[j] * A[j][i];
		}
	}
}
/* a (n x n) square matrix times a vector. */
void matrixprod(double **A, double *a, double *res, const int &n)
{
	for ( int i=0; i<n; i++ ) {
		res[i] = 0;
		for ( int j=0; j<n; j++ ) {
			res[i] += a[j] * A[i][j];
		}
	}
}

void addmatrix(double **src, double **dest, const int &n, double factor)
{
	for (int i=0; i<n; ++i)
		for (int j=0; j<n; ++j)
			src[i][j] += dest[i][j] * factor;
}

void addmatrix(int **dest, int **src, const int &n, const int &m)
{
	for (int i=0; i<n; ++i)
		for (int j=0; j<m; ++j)
			dest[i][j] += src[i][j];
}

void addmatrix(double **src, double **dest, const int &n, const int &m, double factor)
{
	for (int i=0; i<n; ++i)
		for (int j=0; j<m; ++j)
			src[i][j] += dest[i][j] * factor;
}

/* A + ab^\top*/
void addmatrix(double **A, double *a, double *b, const int &n, double factor)
{
	for (int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			A[i][j] += a[i] * b[j] * factor;
		}
	}
}

/* A + ab^\top + ba^\top*/
void addmatrix2(double **A, double *a, double *b, const int &n, double factor)
{
	for (int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			A[i][j] += (a[i] * b[j] + b[i] * a[j]) * factor;
		}
	}
}

/* res = res + a * factor */
void addvec(double *res, double *a, const int &n, double factor)
{
	for ( int i=0; i<n; i++ ) {
		res[i] += a[i] * factor;
	}
}

/* res = res + a */
void addvec(double *res, double *a, const int &n)
{
	for ( int i=0; i<n; i++ ) {
		res[i] += a[i];
	}
}

void vmulti(double *res, const int &n, double factor)
{
	for (int i=0; i<n; ++i) 
		res[i] *= factor;
}

void mmulti(double **res, const int &n, const int &m, double factor)
{
	for (int i=0; i<n; ++i)
		for (int j=0; j<m; ++j)
			res[i][j] *= factor;
}

/* the inverse of a matrix. */
bool inverse(double **A, double **res, const int &n)
{
    ap::real_2d_array a;
    a.setbounds(0, n-1, 0, n-1);

	// upper-triangle matrix
	for ( int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			if ( j < i ) a(i, j) = 0;
			else a(i, j) = A[i][j];
		}
	}

	bool bRes = true;
	// inverse
	if( spdmatrixinverse(a, n, true) ) {
	} else {
		printf("Inverse matrix error!");
		bRes = false;
	}

	for ( int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			if ( j < i ) res[i][j] = a(j, i);
			else res[i][j] = a(i, j);
		}
	}

	return bRes;
}

/* the inverse of a matrix. */
double inverse_det(double **A, double **res, const int &n)
{
    ap::real_2d_array a;
    a.setbounds(0, n-1, 0, n-1);

	// upper-triangle matrix
	for ( int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			if ( j < i ) a(i, j) = 0;
			else a(i, j) = A[i][j];
		}
	}

	double dDet = 1;
    if( spdmatrixcholesky(a, n, true) ) {
		// get cholesky decomposition result. & compute determinant
		for ( int i=0; i<n; i++ ) {
			dDet *= (a(i, i) * a(i, i));
		}

		//for ( int i=0; i<n; i++ ) {
		//	for ( int j=0; j<n; j++ ) {
		//		printf("%.15f  ", a(i, j));
		//	}
		//	printf("\n");
		//}
		//printf("\n");


		// inverse
        if( spdmatrixcholeskyinverse(a, n, true) ) {
        } else {
			printf("Inverse matrix error!");
		}
	} else {
		printf("Non-PSD matrix!");
	}

	for ( int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			if ( j < i ) res[i][j] = a(j, i);
			else res[i][j] = a(i, j);
		}
	}

	return dDet;
}

/* the inverse of a matrix. */
void inverse_cholydec(double **A, double **res, double **lowerTriangle, const int &n)
{
    ap::real_2d_array a;
    a.setbounds(0, n-1, 0, n-1);

	// upper-triangle matrix
	for ( int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			if ( j < i ) a(i, j) = 0;
			else a(i, j) = A[i][j];
		}
	}

    if( spdmatrixcholesky(a, n, true) ) {
		// get cholesky decomposition result.
		for ( int i=0; i<n; i++ ) {
			for ( int j=0; j<n; j++ ) {
				lowerTriangle[i][j] = a(j, i);
			}
		}

		// inverse
        if( spdmatrixcholeskyinverse(a, n, true) ) {
        } else {
			printf("Inverse matrix error!");
		}
	} else {
		printf("Non-PSD matrix!");
	}

	for ( int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			if ( j < i ) res[i][j] = a(j, i);
			else res[i][j] = a(i, j);
		}
	}
}

bool choleskydec(double **A, double **res, const int &n, bool isupper)
{
    ap::real_2d_array a;
    a.setbounds(0, n-1, 0, n-1);
    for (int i=0; i<n; ++i)
	    for (int j=0; j<n; ++j)
		    a(i, j) = A[i][j];

	bool bRes = true;
	if ( !spdmatrixcholesky(a, n, isupper) ) {
		printf("matrix is not positive-definite\n");
		bRes = false;
	}

	if ( isupper ) {
		for ( int i=0; i<n; i++ ) {
			for ( int j=0; j<n; j++ ) {
				if ( j < i ) res[i][j] = 0;
				else res[i][j] = a(i, j);
			}
		}
	} else {
		for ( int i=0; i<n; i++ ) {
			for ( int j=0; j<n; j++ ) {
				if ( j <= i ) res[i][j] = a(i, j);
				else res[i][j] = 0;
			}
		}
	}

	return bRes;
}

double logExpSum(double* eta, int n)
{
	double maxEta = -1e9;
	for (int i=0; i<n; ++i)
		maxEta = max(maxEta, eta[i]);

	double sum = 0;
	for ( int i=0; i<n; ++i )
	{
		double tmp = eta[i]-maxEta;
		if (tmp<-300) tmp = -300;
		sum += exp( tmp );
	}

	// sum < n	
	return maxEta + log(sum);
}

// log( exp(a) + exp(b) )
double logExpSum(double a, double b)
{
	double m = a>b ? a : b;
	a -= m;
	b -= m;
	if (a<-300) a = -300;
	if (b<-300) b = -300;

	return m + log( exp(a)+exp(b) );
}

// log( exp(a) - exp(b) )
double logExpMinus(double a, double b)
{
	double m = a>b ? a : b;
	a -= m;
	b -= m;
	if (a<-300) a = -300;
	if (b<-300) b = -300;

	return m + log( exp(a)-exp(b) );
}

void softmax(double*ret, double* v, int n)
{
	// Numeral safe softmax
	// TODO: improve this O(n^2) procedure
	double sum = 0;
	for (int i=0; i<n; ++i)
	{
		double denom = 1;
		for (int j=0; j<n; ++j)
			if (i!=j)
			{
				double tmp = v[j]-v[i];
				if (tmp>300) denom += exp(300);
				else if (tmp<-300) denom += exp(-300);
				else denom += exp(tmp);
			}
		ret[i] = 1.0 / denom;
	}
}

char* appendNum(const char *prefix, const char *extension, int num)
{
	static char buffer[65536];
	if (num==-1)
		sprintf(buffer, "%sfinal.%s", prefix, extension);
	else
		sprintf(buffer, "%s%d.%s", prefix, num, extension);
	return buffer;
}

char* appendNum(const char *prefix, const char *extension)
{
	static char buffer[65536];
	sprintf(buffer, "%s.%s", prefix, extension);
	return buffer;
}

double perplexity( int D, int K, double **phi, double **theta, int **w, std::vector<int> length )
{
	FILE *tmpFile = fopen("p", "w");
	double logLikelihood = 0;
	int totN = 0;
	for (int d=0; d<D; ++d)
	{
		for (int n=0; n<length[d]; ++n)
		{
			int currentW = w[d][n];
			double p = 0;
			for (int k=0; k<K; ++k)
				p += theta[d][k] * phi[k][currentW];

			if (p < 1e-20)
			{
				printf("Warning: low probability, document = %d\n", d);
				for (int k=0; k<K; ++k)
				{
					printf("theta = %lf\tphi = %lf\n", theta[d][k], phi[k][currentW]);
				}

			    //exit(0);
			}
			fprintf(tmpFile, "%lf\n", log(p));
			logLikelihood += log(p);
		}

		totN += length[d];
	}

	fclose(tmpFile);
	return exp(-logLikelihood / totN);
}

void readCorpus( const char *fileName, int **&w, vector<int> &length, int &D, int &W )
{
	vector<int*> docs;

	FILE *file = fopen(fileName, "r");
	fscanf(file, "%d\n", &W);
	int nitem = 0;
	while (fscanf(file, "%d", &nitem) != EOF)
	{
		vector<int> doc;
		for (int i=0; i<nitem; ++i)
		{
			int a; int t;
			fscanf(file, "%d:%d", &a, &t);
			for (int j=0; j<t; ++j)
			{
				doc.push_back(a);
			}
		}

		int *pdoc = new int[doc.size()];
		for (int i=0; i<doc.size(); ++i)
		{
			pdoc[i] = doc[i];
		}

		docs.push_back(pdoc);
		length.push_back(doc.size());
	}
	fclose(file);

	D = docs.size();
	w = new int*[D];
	for (int i=0; i<D; ++i)
	{
		w[i] = docs[i];	
	}

	printf("Read corpus of %d documents.\n", D);
}

void create_generator(base_generator_type *&generator, uniform_real<> *&uni_dist,
			variate_generator<base_generator_type&, uniform_real<> > *&unif01)
{
   generator = new base_generator_type(time(0));
   uni_dist = new uniform_real<> (0, 1);
   unif01 = new variate_generator<base_generator_type&,
           boost::uniform_real<> > (*(generator), *(uni_dist));
   //Throw away a few initial values
   for (int j = 0; j < 25; j++) {
       (*unif01)();
   }
}

void print(const google::protobuf::Message &message)
{
	string strbuf;
	google::protobuf::TextFormat::PrintToString(message, &strbuf);

	cout << strbuf;
}

int numOfSymmetryMat(int dim)
{
	return (dim * dim - dim ) / 2 + dim;
}

void symmMatToArray(arma::mat m, double *buff)
{
	int dim = m.n_rows;

	register int cnt = 0;
	for (register int c=0; c<dim; ++c)
		for (register int r=0; r<=c; ++r)
			buff[cnt++] = m(r, c);
}

arma::mat arrayToSymmMat(double *buff, int dim)
{
	mat ret(dim, dim);

	register int cnt = 0;
	for (register int c=0; c<dim; ++c)
	{
		for (register int r=0; r<c; ++r)
		{
			ret(r, c) = buff[cnt];
			ret(c, r) = buff[cnt];
			cnt ++;
		}
		ret(c, c) = buff[cnt++];
	}

	return ret;
}

arma::mat MPI_Reduce_Symmetry_Matrix(arma::mat &source, int rank, int size)
{
	int dim = source.n_rows;
	int nelem = numOfSymmetryMat(dim);

	double *send_buff = alloc<double>(nelem);
	double *recv_buff = alloc<double>(nelem);

	symmMatToArray(source, send_buff);

	MPI_Reduce(send_buff, recv_buff, nelem, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	mat ret = rank==0 ? arrayToSymmMat(recv_buff, dim) : mat();

	delete[] send_buff;
	delete[] recv_buff;

	return ret;
}

void MPI_Broadcast_Symmetry_Matrix(arma::mat &work, int rank, int size)
{
	int dim = work.n_rows;
	int nelem = numOfSymmetryMat(dim);

	double *buff = alloc<double>(nelem);

	if (rank==0)
		symmMatToArray(work, buff);

	MPI_Bcast(buff, nelem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank!=0)
		work = arrayToSymmMat(buff, dim);

	delete[] buff;
}
