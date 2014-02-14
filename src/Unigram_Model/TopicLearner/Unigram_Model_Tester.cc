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
 * Unigram_Model_Tester.cpp
 *
 *  Created on: 06-Jan-2011
 *      
 */

#include "Unigram_Model_Tester.h"
#include "document.pb.h"
#include "sampler.h"
#include "Context.h"
#include "glog/logging.h"
#include "TopicLearner/Dirichlet.h"
#include "utils.h"
#include "random.h"
#include "matrixIO.h"
#include <iostream>
using namespace std;
using namespace arma;

Unigram_Model_Tester::Unigram_Model_Tester(TypeTopicCounts& ttc,
        Parameter& alpha, Parameter& beta, mat& mu, mat& cov,
	WordIndexDictionary& dict,
        bool no_init) :
    _ttc(ttc), _alpha(alpha), _beta(beta), _mu(mu), _cov(cov) {
    if (no_init)
        return;
    Context& context = Context::get_instance();
    string input_w = context.get_string("input_w");
    string input_t = context.get_string("input_t");
    string output_t = context.get_string("output_t");
    set_up_io(input_w, output_t);

    _num_words = _ttc.get_num_words();
    _num_topics = _ttc.get_num_topics();

    string ttc_dumpfile = context.get_string("ttc_dumpfile");
    string param_dumpfile = context.get_string("param_dumpfile");

    int num_dumps = context.get_int("numdumps");
    if (num_dumps == 1) {
        LOG(WARNING) << "Initializing Word-Topic counts table from dump "
                << ttc_dumpfile << " using " << _num_words << " words & "
                << _num_topics << " topics.";
        _ttc.initialize_from_dump(ttc_dumpfile, &dict);
    } 
    LOG(WARNING) << "Initialized Word-Topic counts table";

    double alpha_bar = context.get_double("alpha");
    LOG(WARNING) << "Initializing Alpha vector from dumpfile "
            << param_dumpfile;
    _alpha.initialize_from_dump(param_dumpfile);
    LOG(WARNING) << "Alpha vector initialized";

    double beta_flag = context.get_double("beta");
    LOG(WARNING) << "Initializing Beta Parameter from specified Beta = "
            << beta_flag;
    double beta_value[1] = { beta_flag };
    _beta.initialize_from_values(1, beta_value, _num_words * beta_flag);
    LOG(WARNING) << "Beta param initialized";

    // Initialize mu, cov
	string mu_dumpfile = context.get_string("mudump");
	string cov_dumpfile = context.get_string("covdump");
	_mu.load(mu_dumpfile);
	_cov.load(cov_dumpfile);
	_prec = _cov.i();

    ignore_old_topic = false;

	// Debug
	//LOG(WARNING) << "Dictionary";
	//dict.print();

	/*
	LOG(WARNING) << "Output ttc for debug";
	for (int w=0; w<_num_words; ++w)
	{
    	topicCounts current_topic_counts(_num_topics); //Storage for topic counts for the current word
		int *dense_current_topic_counts = alloc<int>(_num_topics);
	    _ttc.get_counts(w, &current_topic_counts);

		for (int j=0; j<current_topic_counts.length; ++j)
		{
			topic_t top = current_topic_counts.items[j].choose.top;
			cnt_t cnt = current_topic_counts.items[j].choose.cnt;

			dense_current_topic_counts[top] += cnt;
		}
		print(dense_current_topic_counts, _num_topics);

		delete[] dense_current_topic_counts;
	}*/
}

Unigram_Model_Tester::~Unigram_Model_Tester() {
    release_io();
}

void Unigram_Model_Tester::set_up_io(string input_w, string output_t) {
    _wdoc_rdr = new DocumentReader(input_w);

    _tdoc_writer = new DocumentWriter(output_t);
}

void Unigram_Model_Tester::release_io() {
    if (_wdoc_rdr)
        delete _wdoc_rdr;
    if (_tdoc_writer)
        delete _tdoc_writer;
}

google::protobuf::Message* Unigram_Model_Tester::allocate_document_buffer(
        size_t num_docs) {
    return new LDA::unigram_document[num_docs];
}

void Unigram_Model_Tester::deallocate_document_buffer(
        google::protobuf::Message* docs) {
    delete[] dynamic_cast<LDA::unigram_document*> (docs);
}

google::protobuf::Message* Unigram_Model_Tester::get_nth_document(
        google::protobuf::Message* docs, size_t n) {
    return dynamic_cast<LDA::unigram_document*> (docs) + n;
}

void* Unigram_Model_Tester::read(google::protobuf::Message& doc) {
    if (_wdoc_rdr->read(&doc) == -1)
        return NULL;
    LDA::unigram_document* wdoc = dynamic_cast<LDA::unigram_document*> (&doc);
    update_t *upd = new update_t;
    upd->doc = wdoc;
    return upd;
}

void* Unigram_Model_Tester::test(void* token) {
	static int processed_documents = 0;
	LOG(WARNING) << "Document " << processed_documents;
	++processed_documents;

    update_t* upd = (update_t*) token;
    LDA::unigram_document& doc = *(upd->doc);
    int num_words_in_doc = doc.body_size();
	upd->change_list = new vector<change_elem_t>();

	// Initialize rng
	base_generator_type *generator;
	uniform_real<> *uni_dist;
	variate_generator<base_generator_type&, uniform_real<> > *unif01;
	create_generator(generator, uni_dist, unif01);
	Random *random = new Random(unif01);

	// Initialize topics
	for (int i=0; i<num_words_in_doc; ++i)
	{
			doc.add_topic_assignment(rand()%_num_topics);
	}
	for (int i=0; i<_num_topics; ++i)
	{
			doc.add_eta(0.0);
	//		doc.add_lambda(0.0);
	}

	// Initialize data structure
	double *prob_buff = alloc<double>(_num_topics + 1);
	double *prob = prob_buff + 1;
	double *ck_denom = alloc<double>(_num_topics);
	double tot_beta = _beta.sum;
	double beta = _beta.values[0];

	int *document_topic_counts = alloc<int>(_num_topics);
    int **new_ttc = alloc2D<int>(_num_words, _num_topics);
	int *new_ck = alloc<int>(_num_topics);

	// Copy global_ttc -> new_ck
    atomic<topic_t> *tokens_per_topic = new atomic<topic_t> [_num_topics]; //Storage for n(t)
	_ttc.get_counts(tokens_per_topic);
	for (int k=0; k<_num_topics; ++k)	
	{
		new_ck[k] = tokens_per_topic[k];
	}
	delete[] tokens_per_topic;

	// Initialize local counts
    for (int n=0; n<num_words_in_doc; ++n)
    {
		++new_ttc[doc.body(n)][doc.topic_assignment(n)];
		++new_ck[doc.topic_assignment(n)];
		++document_topic_counts[doc.topic_assignment(n)];
    }

	for (int k=0; k<_num_topics; ++k)
	{
		ck_denom[k] = 1.0 / (new_ck[k] + tot_beta);
	}

    topicCounts current_topic_counts(_num_topics); //Storage for topic counts for the current word
	int *dense_current_topic_counts = alloc<int>(_num_topics);

    int start_iter = 0;
    int end_iter = Context::get_instance().get_int("iter");

    for (int iter = start_iter; iter < end_iter; ++iter)
    {
		// sample z
		for (int i=0; i<num_words_in_doc; ++i) {	//For each word
			word_t word = doc.body(i);
			LOG_IF(FATAL, word >= _num_words) << "Ahhhhhhhhhhh!";
			topic_t old_topic = doc.topic_assignment(i);
	
			// Compute dense vector of current ttc
	        _ttc.get_counts(word, &current_topic_counts);
			clear(dense_current_topic_counts, _num_topics);
			--dense_current_topic_counts[old_topic];
	
			for (int j=0; j<current_topic_counts.length; ++j)
			{
				topic_t top = current_topic_counts.items[j].choose.top;
				cnt_t cnt = current_topic_counts.items[j].choose.cnt;
	
				dense_current_topic_counts[top] += cnt;
			}

			for (int k=0; k<_num_topics; ++k)
			{
				dense_current_topic_counts[k] += new_ttc[word][k];
			}
	
			document_topic_counts[old_topic]--;		// Local
			new_ck[old_topic]--;			
			ck_denom[old_topic] = 1.0 / (new_ck[old_topic] + tot_beta);
	
			// Compute score
			prob[-1] = 0;
	
			topic_t k;
			for (k=0; k<_num_topics; ++k)
			{
				prob[k] = prob[k-1] +
					// This word among all word from this topic
					(double)(dense_current_topic_counts[k] + beta) * ck_denom[k]
					// Logistic prior ( softmax of mixture coefficients )
	                              * exp(doc.eta(k));
			}
	
			// Draw a random sample
			double sample = (*unif01)() * prob[_num_topics - 1];
	
			for (k = 0; k < _num_topics; ++k)
				if (prob[k] >= sample)
					break;
	
			// It really will cause some trouble if we don't do so
			if (k == _num_topics)
			{
				k = _num_topics-1;
			}
		
			// Update changes
			topic_t new_topic = k;
	
			doc.set_topic_assignment(i, new_topic);
			new_ck[new_topic]++;
			ck_denom[k] = 1.0 / (new_ck[k] + tot_beta);
			document_topic_counts[new_topic]++;
		}

		// sample_eta
		sampleEta(doc, random);
    }

	delete generator;
	delete uni_dist;
	delete unif01;
	delete random;

	delete[] prob_buff;
	delete[] ck_denom;
	delete[] document_topic_counts;
	free2D(new_ttc, _num_words, _num_topics);
	delete[] new_ck;
	delete[] dense_current_topic_counts;

    return upd;
}

void Unigram_Model_Tester::sampleEta(LDA::unigram_document& doc, Random *random)
{
	int num_sub_iter = Context::get_instance().get_int("subiter");
	int *ck = alloc<int>(_num_topics);
	double *eta = alloc<double>(_num_topics);
	double *lambda = alloc<double>(_num_topics);

	// Compute ck
	int doc_length = doc.body_size();
	clear(ck, _num_topics);
	for (int i=0; i<doc_length; ++i)
	{
		ck[doc.topic_assignment(i)]++;
	}

	// Cache eta and lambda
	for (int k=0; k<_num_topics; ++k)
	{
		eta[k] = doc.eta(k);
//		lambda[k] = doc.lambda(k);
	}

	// Sample
	double allExpSum = logExpSum(eta, _num_topics);
	for (int k=0; k<_num_topics; ++k)
	{	
		for (int temp=0; temp<num_sub_iter; ++temp)
		{
			allExpSum = logExpMinus(allExpSum, eta[k]);
	
			double zeta = allExpSum;
								// Swap a[k] and a[T-1] to compute logExpSum(0 ~ T-1)
	
			double rho = eta[k] - zeta;
			lambda[k] = random->nextPG(doc_length, rho);	// lambda_k:	sample from Polya-Gamma
	
			double priorVar = 1.0/_prec(k, k);	// Conditional MVGaussian
			double priorMean = 0;
			for (int a=0; a<_num_topics; ++a)
				if (a!=k)
					priorMean += _prec(a, k) * (eta[a] - _mu(a));
			priorMean = _mu(k) - priorVar*priorMean;
	
			double kappa = ck[k] - (double)doc_length/2;
	
			// Compute conditional distribution with respect to lambda_k
			double tau = 1.0 / ( 1.0 / priorVar + lambda[k] );
			double gamma = tau * (priorMean/priorVar + kappa + lambda[k]*zeta);
	
			eta[k] = random->rnorm()*sqrt(tau) + gamma;//_eta_k: 	sample from normal
			allExpSum = logExpSum(allExpSum, eta[k]);
		}
	}

	// Copy eta and lambda back
	for (int k=0; k<_num_topics; ++k)
	{
		doc.set_eta(k, eta[k]);
//		doc.set_lambda(k, lambda[k]);
	}

	delete[] ck;
	delete[] eta;
	delete[] lambda;
}

void* Unigram_Model_Tester::sample(void*) {
    LOG(FATAL) << "Sample method called for tester";
}

void* Unigram_Model_Tester::update(void*) {
    LOG(FATAL) << "Update method called for tester";
}

void* Unigram_Model_Tester::sampleEta(void*)
{
    LOG(FATAL) << "Sample method called for tester";
}

void* Unigram_Model_Tester::accumulateEta(void*)
{
    LOG(FATAL) << "Accumulate method called for tester";
}

void Unigram_Model_Tester::sampleGauss()
{
    LOG(FATAL) << "Sample Gauss method called for tester";
}

void* Unigram_Model_Tester::optimize(void*) {
    LOG(FATAL) << "Optimize method called for tester";
}

void* Unigram_Model_Tester::eval(void* token, double& eval_value) {
    update_t* upd = (update_t*) token;
    LDA::unigram_document& doc = *(upd->doc);

    double doc_loglikelihood = 0.0;

    topic_t* local_topic_counts = new topic_t[_num_topics];
    topic_t* local_topic_index = new topic_t[_num_topics];
    int num_words_in_doc = doc.body_size();

    /************* Compute Likelihood **************/
    //Clear topic Counts
    memset(local_topic_counts, 0, _num_topics * sizeof(topic_t));
    memset(local_topic_index, 0, _num_topics * sizeof(topic_t));

    int non_zero_topics = 0;
    topic_t* ltcInd = local_topic_counts;
    topic_t* ltdInd = local_topic_index;
    if (!ignore_old_topic) {
        //Build the dense local_topic_counts with index
        for (int k = 0; k < num_words_in_doc; k++) {
            topic_t topic = doc.topic_assignment(k);
            ++local_topic_counts[topic];
        }

        for (topic_t t = 0; t < _num_topics; t++) {
            int n = *ltcInd;
            if (n != 0) {
                *ltdInd = t;
                ++ltdInd;
                ++non_zero_topics;
            }
            ++ltcInd;
        }
        //----- Local Topic Counts built
    }

    //Accumulate loglikelihood for this doc
    ltdInd = local_topic_index;
    for (int i = 0; i < non_zero_topics; i++) {
        topic_t topic = *ltdInd;
        double gal = _alpha.values[topic];
        int cnt = local_topic_counts[topic];
        doc_loglikelihood += log_gamma(gal + cnt) - log_gamma(gal);

        LOG_IF(FATAL,std::isnan(doc_loglikelihood))<< gal << "," << cnt << "," <<_alpha.sum << "," << num_words_in_doc;

        ++ltdInd;
    }
    doc_loglikelihood += log_gamma(_alpha.sum) - log_gamma(_alpha.sum + num_words_in_doc);
    /************* Compute Likelihood **************/

    delete [] local_topic_counts;
    delete [] local_topic_index;

    eval_value = doc_loglikelihood;
    return upd;
}

void Unigram_Model_Tester::write(void* token) {
    update_t* upd = (update_t*) token;
    LDA::unigram_document& doc = *(upd->doc);
    doc.clear_body();
    LOG_IF(FATAL, !_tdoc_writer->write(doc) )<< "Couldn't write to the stream. Quitting..." << endl;
    delete upd;
}

void Unigram_Model_Tester::iteration_done() {
    release_io();
    Context& context = Context::get_instance();
    string input_w = context.get_string("input_w");
    string input_t = context.get_string("input_t");
    string output_t = context.get_string("output_t");

    string cmd = "cp " + output_t + " " + input_t;
    system(cmd.c_str());
}
