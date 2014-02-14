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
 * Unigram_Model_Builder.cpp
 *
 *  Created on: 06-Jan-2011
 *      
 */

#include <mpi.h>
#include "Unigram_Model_Testing_Builder.h"
#include "Context.h"
#include "DocumentReader.h"
#include "DocumentWriter.h"
#include "Unigram_Model_Tester.h"
#include "TopicLearner/TBB_Pipeline.h"
#include "TopicLearner/Testing_Execution_Strategy.h"
#include <armadillo>
using namespace arma;

Unigram_Model_Testing_Builder::Unigram_Model_Testing_Builder() {
}

Unigram_Model_Testing_Builder::~Unigram_Model_Testing_Builder() {
}

Model_Refiner& Unigram_Model_Testing_Builder::create_model_refiner() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ostringstream sout;
    sout << "." << rank;
    string dotrank = sout.str();

    Context& context = Context::get_instance();
    string input_prefix = context.get_string("inputprefix");
    string input_w = input_prefix + ".wor" + dotrank;
    int num_topics = context.get_int("topics");

    context.put_string("input_w", input_w);

    //The following is needed so that the create_output
    //method which is not overridded works. It uses input_t
    //as the topic assignments file and in testing we do
    //not need that. It also deletes the output_t file.
    //So we switch back to creating a tmp out file
    string input_t = input_prefix + ".top" + dotrank;
    context.put_string("input_t", input_t);
    string output_t = input_prefix + ".top.tmp" + dotrank;
    context.put_string("output_t", output_t);

    string ttc_dumpfile = context.get_string("dumpprefix") + ".ttc.dump";
    string param_dumpfile = context.get_string("dumpprefix") + ".par.dump";
    string global_dict_dump = context.get_string("inputprefix")
            + ".dict.dump.global";

    context.put_string("ttc_dumpfile", ttc_dumpfile);
    context.put_string("param_dumpfile", param_dumpfile);
    context.put_string("global_dict_dump", global_dict_dump);

    string mudump = context.get_string("dumpprefix") + ".mu" + ".dump";
    string covdump = context.get_string("dumpprefix") + ".cov" + ".dump";
    context.put_string("mudump", mudump);
    context.put_string("covdump", covdump);

    TypeTopicCounts& ttc = _model->get_ttc();
    Parameter& alpha = _model->get_parameter(Unigram_Model::ALPHA);
    Parameter& beta = _model->get_parameter(Unigram_Model::BETA);
	mat& mu = _model->get_mat(Unigram_Model::MU);
	mat& cov = _model->get_mat(Unigram_Model::COV);

    _refiner = new Unigram_Model_Tester(ttc, alpha, beta, mu, cov, get_dict());
    return *_refiner;
}

Pipeline& Unigram_Model_Testing_Builder::create_pipeline(
        Model_Refiner& refiner) {
    _pipeline = new TBB_Pipeline(refiner);
    return *_pipeline;
}

Execution_Strategy& Unigram_Model_Testing_Builder::create_execution_strategy(
        Pipeline& pipeline) {
    _strategy = new Testing_Execution_Strategy(pipeline, *_model);
    return *_strategy;
}
