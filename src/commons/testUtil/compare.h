#ifndef __COMPARE
#define __COMPARE

class Parameter;

double norm(const Parameter& parameter);

double meanTest(int n, const Parameter *samples, const Parameter& expectedMean);

double averageDifferenceMatrixRows(double **expected,
			double **realvalue,
			int nrows, int ncols);

void bestPermutation(double **Phi, double **inf_Phi, int *perm, int K, int W);

#endif
