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
 * PThread_Pipeline.h
 *
 *
 *  Created on: 14-May-2013
 *      
 */

#ifndef PThread_PIPELINE_H_
#define PThread_PIPELINE_H_

#include "Pipeline.h"
#include "Unigram_Model/TopicLearner/Unigram_Model_Trainer.h"
#include "types.h"
#include <vector>
#include "tbb/atomic.h"
#include "tbb/spin_mutex.h"

//!An implementation of the Pipeline interface using
//!PThread. It is only a dummy pipeline to use yahoo-lda's interface.
class PThread_Pipeline: public Pipeline {
public:
    PThread_Pipeline(Unigram_Model_Trainer&);
    virtual ~PThread_Pipeline();
    void init();
    void add_reader();
    void add_sampler();
    void add_updater();
    void add_etasampler();
    void add_accumulator();
    void add_gausssampler();
    void add_optimizer();
    void add_eval();
    void add_writer();
    void add_tester();
    void clear();
    void destroy();
    void run();
    void dump(int lag);

    Model_Refiner& get_refiner();
    double get_eval();

public:
    void readNwrite(void);
    void sample(int thread_id);

protected:
    Unigram_Model_Trainer& _refiner;
    std::vector<update_t*> *_readSet, *_writeSet, *_workingSet;

    bool _ifSampleTopic;
    bool _ifUpdate;
    bool _ifSampleEta;
    bool _ifAccumulate;
    bool _ifSampleGauss;
    bool _ifEval;
    int _num_threads;

    // count down
    tbb::atomic<int> _current_document;
    tbb::atomic<int> _eval_words;
    double _eval_likelihood;
    tbb::spin_mutex _eval_likelihood_m;

	double time_sample;
	double time_gauss;
	double time_accumulate;
};

#endif /* PThread_PIPELINE_H_ */
