#include "compare.h"
#include <cmath>
#include "../TopicLearner/Parameter.h"
#include "glog/logging.h"
#include <cstdio>
#include "matrixIO.h"
#include "utils.h"

double meanTest(int n, const Parameter *samples, const Parameter& expectedMean)
{
	if (n>0)
		LOG_IF(FATAL, samples[0].length!=expectedMean.length)
				<< "vector dimensions must agree";

	// norm(samples-expectedMean) / norm(samples+expectedMean)
	Parameter sampleMean;
	sampleMean.initialize_from_values(expectedMean.length, NULL, 0);
	for (int i=0; i<n; ++i)
			sampleMean = sampleMean + samples[i];

	sampleMean = sampleMean * (1.0/n);

	return (sampleMean-expectedMean).norm() / (sampleMean+expectedMean).norm();
}

double averageDifferenceMatrixRows(double **expected,
			double **realvalue,
			int nrows, int ncols)
{
	double error = 0;
	for (int i=0; i<nrows; ++i)
	{
		Parameter p_realvalue, p_expected;
		p_expected.initialize_from_values(ncols, expected[i]);
		p_realvalue.initialize_from_values(ncols, realvalue[i]);

		error += meanTest(1, &p_realvalue, p_expected);
	}

	return error / nrows;
}

void bestPermutation(double **cost, int *perm, int *bestPerm, bool *used, 
							   int i, double currentCost, double &bestCost, int K)
{
	if (currentCost > bestCost)
		return;
	if (i==K)
	{
		bestCost = currentCost;
		memcpy(bestPerm, perm, sizeof(int)*K);
		return;
	}
	for (int j=0; j<K; ++j)
		if (!used[j])
		{
			used[j] = true;
			perm[i] = j;
			bestPermutation(cost, perm, bestPerm, used, 
				i+1, currentCost+cost[i][j], bestCost, K);
			used[j] = false;
		}
}

void bestPermutation(double **Phi, double **inf_Phi, int *perm, int K, int W)
{
	bool *used = new bool[K];
	memset(used, 0, K*sizeof(bool));
	double **cost = alloc2D<double>(K, K);
	for (int i=0; i<K; ++i)
		for (int j=0; j<K; ++j)
			for (int w=0; w<W; ++w)
				cost[i][j] += fabs(Phi[i][w] - inf_Phi[j][w]);

	int *temp_perm = new int[K];
	double bestCost = 1e9;

	bestPermutation(cost, temp_perm, perm, used, 0, 0, bestCost, K);

	delete[] used;
	free2D<double>(cost, K, K);
	delete[] temp_perm;
}
