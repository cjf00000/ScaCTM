#ifndef __BIT_H
#define __BIT_H

#include <cstdio>

// binary indexed tree
class BIT
{
public:
	BIT(int size);
	~BIT();

	void initialize(double *values);

	void update(int index, double new_value)
	{
		int n = index + 1;
	
		double delta = new_value - _values[n];
		_values[n] = new_value;
		
		sum += delta;
		// TODO update BIT, O(1)-->O(log n)
	}

	int upper_bound_sum(double x) const
	{
		// TODO search BIT, O(n)-->O(log n)
		double sum = 0;
		for (int i=1; i<=_size; ++i)
		{
			sum += _values[i];

			if (sum >= x)
				return i-1;
		}

		return _size-1;
	}

	double *values;
	double sum;

private:
	double *_bit;
	double *_values;
	double *__bit;
	int _size;
	int _capacity;
};

#endif
