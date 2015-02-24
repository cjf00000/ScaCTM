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
 * Simple_Execution_Strategy.cpp
 *
 *  Created on: 28-Dec-2010
 *      
 */

#include <mpi.h>
#include <cstdlib>
#include "Training_Execution_Strategy.h"
#include "Context.h"
#include "tbb/tick_count.h"
#include "glog/logging.h"

Training_Execution_Strategy::Training_Execution_Strategy(Pipeline& pipeline,
        Model& model, Checkpointer& checkpointer) :
    _pipeline(pipeline), _model(model), _checkpointer(checkpointer) {
}

Training_Execution_Strategy::~Training_Execution_Strategy() {
}

void Training_Execution_Strategy::execute() {
    Context& context = Context::get_instance();
    int start_iter = context.get_int("startiter");
    int end_iter = context.get_int("iter");
    int loglikelihood_interval = context.get_int("printloglikelihood");
    int optimize_interval = context.get_int("optimizestats");
    int burnin = context.get_int("burnin");
    int chkpt_interval = context.get_int("chkptinterval");

    //Check if checkpoint metadata is available
    std::string prefix = context.get_string("inputprefix");
    context.put_string("chkpt_file", prefix + ".chk");
    std::string state = _checkpointer.load_metadata();
    if (state.size() > 0)
        start_iter = *((int*) state.c_str()) + 1;

    LOG(WARNING) << "Starting Parallel training Pipeline";

    double t_start = MPI_Wtime();
    int lag = context.get_int("lag");
	LOG(WARNING) << "lag is " << lag;
    for (int iter = start_iter; iter <= end_iter; ++iter) {
        bool compute_loglikelihood = (iter == start_iter) || (iter
                % loglikelihood_interval == 0);
        bool optimize = (iter > burnin) && (iter % optimize_interval == 0);
        _pipeline.init();
        _pipeline.add_reader();
	if (!Context::get_instance().get_bool("skipiniteta") && !Context::get_instance().get_bool("skipinitz"))
	{
	        _pipeline.add_sampler();
        	_pipeline.add_updater();
	}

	if (!Context::get_instance().get_bool("skipiniteta") && !Context::get_instance().get_bool("testml"))
	{
		_pipeline.add_etasampler();
	}

	if (!Context::get_instance().get_bool("skipinitz") && !Context::get_instance().get_bool("testml"))
	{
	    _pipeline.add_accumulator();
	    _pipeline.add_gausssampler();
	}

	if (compute_loglikelihood)
	{
		_pipeline.add_eval();
	}

        _pipeline.add_writer();
        tbb::tick_count t0 = tbb::tick_count::now();
        _pipeline.run();
        tbb::tick_count t1 = tbb::tick_count::now();
        LOG(WARNING) << "Iteration " << iter << " done. Took "
                << (t1 - t0).seconds() / 60 << " mins" << endl;
    	double t_end = MPI_Wtime();
		LOG(WARNING) << "Time elapsed: " << t_end - t_start;

        if (compute_loglikelihood) {
            double doc_loglikelihood = _pipeline.get_eval();
            LOG(WARNING)
                    << ">>>>>>>>>> Perplexity: " << doc_loglikelihood;
        }
	if (lag!=-1 && iter%lag==0)
	{
		_pipeline.dump(iter);
	}

        _pipeline.clear();
        _pipeline.destroy();
        if (iter % chkpt_interval == 0) {
            LOG(WARNING) << ">>>>>>>>>>> Check Pointing at iteration: "
                    << iter;
            std::string chkpt_state((char*) &iter, sizeof(int));
            _checkpointer.save_metadata(chkpt_state);
            _checkpointer.checkpoint();
        }
    }

    LOG(WARNING) << "Parallel training Pipeline done";

    double t_end = MPI_Wtime();
    LOG(WARNING) << "Time elapsed: " << t_end - t_start;
}
