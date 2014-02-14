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
 * PThread_Pipeline.cpp
 *
 *  Created on: 04-Jan-2011
 *      
 */

#include "PThread_Pipeline.h"
#include "Context.h"
#include <pthread.h>
#include "glog/logging.h"
#include <armadillo>
using namespace std;
using namespace arma;

PThread_Pipeline::PThread_Pipeline(Unigram_Model_Trainer& refiner) :
    _refiner(refiner) {
    _num_threads = Context::get_instance().get_int("samplerthreads");
	time_gauss = 0;
	time_sample = 0;
	time_accumulate = 0;
}

PThread_Pipeline::~PThread_Pipeline() 
{ 
}

void PThread_Pipeline::init() {
    _ifSampleGauss = false;
    _ifUpdate = false;
    _ifAccumulate = false;
    _ifSampleEta = false;
    _ifSampleTopic = false;

	_readSet = new vector<update_t*>();
	_writeSet = new vector<update_t*>();
	_workingSet = new vector<update_t*>();

	int miniBatchSize = Context::get_instance().get_int("minibatchsize");
	_readSet->reserve(miniBatchSize);
	_writeSet->reserve(miniBatchSize);
	_workingSet->reserve(miniBatchSize);
}

void PThread_Pipeline::destroy() {
	delete _readSet;
	delete _writeSet;
	delete _workingSet;

	LOG(WARNING) << "Sample took " << time_sample << " seconds";
	LOG(WARNING) << "Accumulate took " << time_accumulate << " seconds";
	LOG(WARNING) << "Gauss took " << time_gauss << " seconds";
}

void PThread_Pipeline::clear() {
    _refiner.iteration_done();
}

void PThread_Pipeline::add_reader() {
	
}

void PThread_Pipeline::add_sampler() {
	_ifSampleTopic = true;
}

void PThread_Pipeline::add_updater() {
	_ifUpdate = true;
}

void PThread_Pipeline::add_optimizer() {
	LOG(FATAL) << "Unsupported method add_optimizer";
}

void PThread_Pipeline::add_etasampler()
{
	_ifSampleEta = true;
}

void PThread_Pipeline::add_accumulator() {
	_ifAccumulate = true;
}

void PThread_Pipeline::add_gausssampler()
{
	_ifSampleGauss = true;
}

void PThread_Pipeline::add_eval() {
	LOG(FATAL) << "Unsupported method add_eval";
}

void PThread_Pipeline::add_writer() {

}

void PThread_Pipeline::add_tester() {
	LOG(FATAL) << "Unsupported method add_tester";
}

Model_Refiner& PThread_Pipeline::get_refiner() {
    return _refiner;
}

double PThread_Pipeline::get_eval() {
	LOG(FATAL) << "Unsupported method get_eval";
	
	return 0;
}

struct ThreadObject
{
	PThread_Pipeline* source;
	int thread_id;
};

void* thread_readNwrite(void *data)
{
	ThreadObject* obj = (ThreadObject*)data;

	obj->source->readNwrite();

	pthread_exit(NULL);
}

void PThread_Pipeline::readNwrite(void)
{
	int miniBatchSize = Context::get_instance().get_int("minibatchsize");

	// Read
	while (_readSet->size() < miniBatchSize)
	{
		update_t* upd = (update_t*)_refiner.pthread_read();
		if (upd==NULL)
		{
			break;
		}

		_readSet->push_back(upd);
	}

	// Write
	for (int i=0; i<_writeSet->size(); ++i)
	{
		update_t* upd = (*_writeSet)[i];

		// I am not responsible for deleting upd
		_refiner.pthread_write(upd);
	}
	_writeSet->clear();
}

void* thread_sample(void *data)
{
	ThreadObject* obj = (ThreadObject*)data;

	obj->source->sample(obj->thread_id);

	pthread_exit(NULL);
}

void PThread_Pipeline::sample(int thread_id)
{
	while (1)
	{
		int my_current_document = _current_document.fetch_and_decrement();

		if (my_current_document < 0)
		{
			break;
		}

		update_t *upd = (*_workingSet)[my_current_document];

		if (_ifSampleTopic)
			_refiner.sample(upd, thread_id);

		if (_ifUpdate)
			_refiner.update(upd);

		if (_ifSampleEta)
			_refiner.sampleEta(upd, thread_id);
	}
}

void PThread_Pipeline::run() {
	LOG(WARNING) << "Iteration start. readset = " << _readSet->size()
			<< ", writeset = " << _writeSet->size() << ", workingset = " << _workingSet->size();

	wall_clock myclock;
	do{
		_current_document = _workingSet->size() - 1;

		myclock.tic();

		// Create read and write thread
		pthread_t readNwrite_thread;
		ThreadObject o_readNwrite;
		o_readNwrite.source = this;
		o_readNwrite.thread_id = -1;
		pthread_create(&readNwrite_thread, NULL, thread_readNwrite, &o_readNwrite);

		// Create sample threads
		int samplerThreads = Context::get_instance().get_int("samplerthreads");
                LOG(WARNING) << "!!!!!!! samplerThreads: " << samplerThreads;
		pthread_t *sample_thread = new pthread_t[samplerThreads];
		ThreadObject *o_sample = new ThreadObject[samplerThreads];
		for (int i=0; i<samplerThreads; ++i)
		{
			o_sample[i].source = this;
			o_sample[i].thread_id = i;
			pthread_create(&sample_thread[i], NULL, thread_sample, &o_sample[i]);
		}

		// Join threads
		pthread_join(readNwrite_thread, NULL);
		for (int i=0; i<samplerThreads; ++i)
		{
			pthread_join(sample_thread[i], NULL);
		}

		delete[] sample_thread;
		delete[] o_sample;

		time_sample += myclock.toc();

		// Accumulate
		myclock.tic();
		if (_ifAccumulate)
		{
			_refiner.pthread_accumulateEta(*_workingSet);
		}
		time_accumulate += myclock.toc();

		LOG_IF(FATAL, !_writeSet->empty())
				<< "Write set is not empty" << endl;

		vector<update_t*> *tmp;
		tmp = _writeSet;
		_writeSet = _workingSet;
		_workingSet = _readSet;
		_readSet = tmp;

		LOG(WARNING) << "Mini batch done. readset = " << _readSet->size()
				<< ", writeset = " << _writeSet->size() << ", workingset = " << _workingSet->size();
	}
	while (!_readSet->empty() || !_writeSet->empty() || !_workingSet->empty());

	myclock.tic();
    if (_ifSampleGauss)
    {
	    _refiner.sampleGauss();
    }
	time_gauss += myclock.toc();
}

void PThread_Pipeline::dump(int lag)
{
    _refiner.dump(lag);
}
