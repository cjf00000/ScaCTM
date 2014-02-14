/*******************************************************************************
    Copyright (c) 2011 Yahoo! Inc. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License. See accompanying LICENSE file.

    The Initial Developer of the Original Code is Shravan Narayanamurthy.
******************************************************************************/
/*
 * Parameter.h
 *
 *
 *
 *  Created on: 14-May-2010
 *      
 */

#ifndef PARAMETER_H_
#define PARAMETER_H_

#include <string>
#include <iostream>

using namespace std;

/**
 * A class to represent the various parameters like
 * the dirichlet conjugate weights and such
 *
 * Can be vector valued parameters
 *
 * This class additionally stores the sum and provides
 * functionality to dump to and initialize from disk
 */
typedef struct Parameter {
    int length;
    double* values;
    double sum;

    Parameter();
    Parameter(const Parameter &param);
    virtual ~Parameter();

    Parameter& operator=(const Parameter &param);
    virtual void dump(string fname);
    virtual void initialize_from_dump(string fname);
    virtual void initialize_from_values(int length_, const double* values);
    virtual void initialize_from_values(int length_, const double* values_,
            float sum_);
	double norm() const;

	// If the parameter is an unrolled matrix, use this 
	// to get a pointer to a matrix to manipulate it.
	// ==============CAUTION==============
	// It is user's duty to delete this handler, delete[] handler;
	virtual double** getMatrixHandler(int rows, int cols);
	friend ostream& operator << (ostream& out, const Parameter& param);
} param;

Parameter operator + (const Parameter& a, const Parameter& b);
Parameter operator - (const Parameter& a, const Parameter& b);
Parameter operator * (const Parameter& a, double factor);
bool operator == (const Parameter &a, const Parameter &b);
bool operator != (const Parameter &a, const Parameter &b);


#endif /* PARAMETER_H_ */
