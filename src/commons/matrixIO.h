#ifndef __MATRIX_IO
#define __MATRIX_IO
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
using std::ifstream;
using std::ofstream;
using std::endl;
using std::cout;
using std::setw;
using std::setiosflags;

template<class T>
void save(const char* file, T *v, int n, double factor = 1)
{
	ofstream fout(file);
	for (int i=0; i<n; ++i)
		fout << (T)(v[i]*factor) << ' ';	
	fout << endl;
}

template<class T>
void save(const char* file, T **m, int r, int c, double factor = 1)
{
	ofstream fout(file);
	for (int i=0; i<r; ++i)
	{
		for (int j=0; j<c; ++j)
			fout << (T)(m[i][j]*factor) << ' ';
		fout << '\n';
	}
}

template<class T>
void load(T* res, const char* file, int n)
{
	ifstream fin(file);
	for (int i=0; i<n; ++i)
		fin >> res[i];
}

template<class T>
void load(T** res, const char* file, int n, int m)
{
	ifstream fin(file);
	for (int i=0; i<n; ++i)
		for (int j=0; j<m; ++j)
			fin >> res[i][j];
}

template<class T>
void print(T *a, int n)
{
	for (int i=0; i<n; ++i)
		cout << setiosflags(std::ios::fixed) << setw(5) << a[i] << '\t';
	cout << endl;
}

template<class T>
void print(T **a, int n, int m)
{
	for (int i=0; i<n; ++i)
	{
		for (int j=0; j<m; ++j)
			cout << setiosflags(std::ios::fixed) << setw(5) << a[i][j] << '\t';
		cout << endl;
	}
}

#endif
