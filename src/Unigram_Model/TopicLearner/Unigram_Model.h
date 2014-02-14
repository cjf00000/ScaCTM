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
 * Unigram_Model.h
 *
 *  Created on: 05-Jan-2011
 *      
 */

#ifndef UNIGRAM_MODEL_H_
#define UNIGRAM_MODEL_H_

#include "TopicLearner/Model.h"
#include <string>
#include "TopicLearner/Parameter.h"
#include "TypeTopicCounts.h"
#include "TopicLearner/GenericTopKList.h"
#include <armadillo>

using namespace std;

class Unigram_Model: public Model {
public:
    const static int ALPHA = 1;
    const static int BETA = 2;
    const static int MU_0 = 3;
    const static int RHO = 4;
	const static int WISHART = 5;
    const static int MU = 6;
	const static int COV = 7;

public:
    Unigram_Model(int, int);
    virtual ~Unigram_Model();

    Parameter& get_parameter(int);
	arma::mat& get_mat(int);
    void set_parameter(int, Parameter&);
    void set_parameter(int, arma::mat&);
    int& get_kappa();

    TypeTopicCounts& get_ttc();

    double get_eval();

    bool save();

    void write_statistics(WordIndexDictionary&);

private:
    TypeTopicCounts* _ttc;
    param _alpha, _beta;

	arma::mat _mu_0;
    param _rho;
	arma::mat _wishart;
    int _kappa;

	arma::mat _mu;
	arma::mat _cov;

    typedef GenericTopKList<wppair, wppair_gt> topK_word_prop_t;
    topK_word_prop_t** top_words_per_topic;
    bool _top_words_empty;

    int _num_words;
    int _num_topics;
};

#endif /* UNIGRAM_MODEL_H_ */
