#include "BIT.h"
#include <cstdio>
#include <memory.h>

BIT::BIT(int size)
{
	_size = size;
	_capacity = 1;
	while (_capacity < _size)
	{
		_capacity *= 2;
	}

	// Start from 0
	__bit = new double[_capacity];
	values = new double[_capacity];

	// Start from 1
	_bit = __bit - 1;
	_values = values - 1;
	memset(__bit, 0, sizeof(double)*_capacity);
	memset(values, 0, sizeof(double)*_capacity);
	sum = 0;
}

BIT::~BIT()
{
	delete[] values;
	delete[] __bit;
}

void BIT::initialize(double *values)
{
	// From 0
	for (int i=0; i<_size; ++i)
	{
		this->values[i] = values[i];
		for (int j=i+1; j<=_capacity; j+=(j&-j))
			_bit[j] += values[i];
	}

	for (int i=_size; i<_capacity; ++i)
	{
		this->values[i] = 0;
	}

	sum = _bit[_capacity];
}

