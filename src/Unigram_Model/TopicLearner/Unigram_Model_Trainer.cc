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
 * Unigram_Model_Trainer.cpp
 *
 *  Created on: 06-Jan-2011
 *      
 */

#include <mpi.h>
#include "Unigram_Model_Trainer.h"
#include "document.pb.h"
#include "Context.h"
#include "glog/logging.h"
#include "TopicLearner/Dirichlet.h"
#include "utils.h"
#include "random.h"
#include "matrixIO.h"
#include "BIT.h"
#include <sstream>
using namespace arma;

Unigram_Model_Trainer::Unigram_Model_Trainer(TypeTopicCounts& ttc,
        Parameter& alpha, Parameter& beta, mat& mu_0, Parameter& rho,
	int &kappa, mat& wishart,
	mat& mu, mat& cov) :
    _ttc(ttc), _alpha(alpha), _beta(beta), _mu_0(mu_0), _rho(rho), _kappa(kappa), _wishart(wishart),
    _mu(mu), _cov(cov){
    Context& context = Context::get_instance();
    ignore_old_topic = context.get_bool("ignore_old_topic");
    string input_w = context.get_string("input_w");
    string input_t = context.get_string("input_t");
    string output_t = context.get_string("output_t");
    set_up_io(input_w, input_t, output_t);

    _num_words = _ttc.get_num_words();
    _num_topics = _ttc.get_num_topics();
    if(ignore_old_topic){
        LOG(WARNING) << "Skipping Word-Topic counts table initialization "
                     << "as online mode has been selected";
    }
    else{
        LOG(WARNING) << "Initializing Word-Topic counts table from docs "
                << input_w << ", " << input_t << " using " << _num_words
                << " words & " << _num_topics << " topics.";
        _ttc.initialize_from_docs(input_w, input_t);
        LOG(WARNING) << "Initialized Word-Topic counts table";
    }

    double alpha_bar = context.get_double("alpha");
    LOG(WARNING) << "Initializing Alpha vector from Alpha_bar = "
            << alpha_bar;
    _alpha.initialize_from_values(_num_topics, NULL, alpha_bar);
    LOG(WARNING) << "Alpha vector initialized";

    double beta_flag = context.get_double("beta");
    LOG(WARNING) << "Initializing Beta Parameter from specified Beta = "
            << beta_flag;
    double beta_value[1] = { beta_flag };
    _beta.initialize_from_values(1, beta_value, _num_words * beta_flag);
    LOG(WARNING) << "Beta param initialized";

    // NIW
    int prior = context.get_int("prior");
    double rho_flag = prior;
    LOG(WARNING) << "Initializing Normal-Inverse-Wishart Prior from specified Rho = "
	    << rho_flag << " Prior = "<< prior;
    _rho.initialize_from_values(1, &rho_flag, rho_flag);
    _kappa = prior;

	_mu_0 = zeros<mat>(_num_topics, 1);
	_wishart = prior * eye<mat>(_num_topics, _num_topics);

    // Normal
    LOG(WARNING) << "Initializing Normal distribution...";

	if (!Context::get_instance().get_bool("skipinitz"))
	{
		_mu = zeros<mat>(_num_topics, 1);
		_cov = eye<mat>(_num_topics, _num_topics);
	}
	else
	{
		string mu_dumpfile = context.get_string("mudump");
		string cov_dumpfile = context.get_string("covdump");
		_mu.load(mu_dumpfile);
		_cov.load(cov_dumpfile);
	}

	_prec = _cov.i();

    // For Accumulate Eta
	mini_batch_size = context.get_int("minibatchsize");
	LOG(WARNING) << "Mini batch size is " << mini_batch_size << endl;

	_current_num_eta = 0;
    _num_documents = 0;

	_eta_s.zeros(_num_topics);
	_q_s.zeros(_num_topics, _num_topics);
	_eta_buff.zeros(_num_topics, mini_batch_size);

    //Initialize the boost RNG
	create_generator(generatorForGauss, uni_distForGauss, unif01ForGauss);
	_randomForGauss = new Random(unif01ForGauss);
    for (int i = 0; i < NUM_RNGS; i++) {
		create_generator(generators[i], uni_dists[i], unif01[i]);

		create_generator(generatorsForEta[i], uni_distsForEta[i], unif01ForEta[i]);
		_randomsForEta[i] = new Random(unif01ForEta[i]);
    }
    rng_ind = 0;
	rng_indForEta = 0;

    //For optimizer
    tau = 100;
    part_grads = new double [_num_topics];
    memset(part_grads,0,_num_topics*sizeof(double));
    part_grads_top_indep = -1 * tau * digamma(alpha.sum);
}

Unigram_Model_Trainer::~Unigram_Model_Trainer() {
	delete generatorForGauss;
	delete uni_distForGauss;
	delete unif01ForGauss;
	delete _randomForGauss;

    for (int i = 0; i < NUM_RNGS; i++) {
        delete generators[i];
        delete uni_dists[i];
        delete unif01[i];

		delete generatorsForEta[i];
		delete uni_distsForEta[i];
		delete unif01ForEta[i];

		delete _randomsForEta[i];
    }

    release_io();
}

void Unigram_Model_Trainer::set_up_io(string input_w, string input_t,
        string output_t) {
    _wdoc_rdr = new DocumentReader(input_w);

    _tdoc_rdr = (ignore_old_topic) ? NULL : new DocumentReader(input_t);

    _tdoc_writer = new DocumentWriter(output_t);
}

void Unigram_Model_Trainer::release_io() {
    delete _wdoc_rdr;
    if(_tdoc_rdr!=NULL) delete _tdoc_rdr;
    else ignore_old_topic = false;
    delete _tdoc_writer;
}

google::protobuf::Message* Unigram_Model_Trainer::allocate_document_buffer(
        size_t num_docs) {
    return new LDA::unigram_document[num_docs];
}

void Unigram_Model_Trainer::deallocate_document_buffer(
        google::protobuf::Message* docs) {
    delete[] dynamic_cast<LDA::unigram_document*> (docs);
}

google::protobuf::Message* Unigram_Model_Trainer::get_nth_document(
        google::protobuf::Message* docs, size_t n) {
    return dynamic_cast<LDA::unigram_document*> (docs) + n;
}

void* Unigram_Model_Trainer::read(google::protobuf::Message& doc) {
    if (_wdoc_rdr->read(&doc) == -1)
        return NULL;
    LDA::unigram_document* wdoc = dynamic_cast<LDA::unigram_document*> (&doc);
    if(!ignore_old_topic){
        LDA::unigram_document tdoc;
        _tdoc_rdr->read(&tdoc);
        wdoc->MergeFrom(tdoc);
    }
    update_t *upd = new update_t;
    upd->doc = wdoc;
    return upd;
}

void* Unigram_Model_Trainer::pthread_read()
{
	LDA::unigram_document *wdoc = new LDA::unigram_document();
	LDA::unigram_document tdoc;

	if (_wdoc_rdr->read(wdoc) == -1)
	{
		delete wdoc;
		return NULL;
	}

	_tdoc_rdr->read(&tdoc);
	wdoc->MergeFrom(tdoc);

	update_t *upd = new update_t;
	upd->doc = wdoc;
	return upd;
}

void* Unigram_Model_Trainer::sample(void* token) {
	LOG(FATAL) << "Deprecated method" << endl;
    update_t* upd = (update_t*) (token);
    vector<change_elem_t> *updates = 0;
    updates = new vector<change_elem_t> ();
    updates->reserve(EXP_NUM_WORDS_PER_DOC);
	//Deprecated
    sample_topics(upd, updates, *(unif01[0]));
    return upd;
}

void* Unigram_Model_Trainer::sample(void* token, int thread_id) {
    update_t* upd = (update_t*) (token);
    vector<change_elem_t> *updates = 0;
    updates = new vector<change_elem_t> ();
    updates->reserve(EXP_NUM_WORDS_PER_DOC);
    sample_topics(upd, updates, *(unif01[thread_id]));
    return upd;
}

void Unigram_Model_Trainer::sample_topics(update_t* upd, vector<change_elem_t> *updates,
		variate_generator<base_generator_type&, boost::uniform_real<> > &unif)
{
    LDA::unigram_document* doc = upd->doc;
    int num_words_in_doc = doc->body_size();

	double *prob_buff = alloc<double>(_num_topics + 1);
	double *prob = prob_buff + 1;
	double beta_sum = _beta.sum;
	double beta = _beta.values[0];
	int non_zero_topics = 0;

	int *document_topic_counts = alloc<int>(_num_topics);
	for (int i=0; i<num_words_in_doc; ++i)
	{
		++document_topic_counts[doc->topic_assignment(i)];
	}
	for (int i=0; i<_num_topics; ++i)
	{
		non_zero_topics += (document_topic_counts[i]>0);
	}

    tbb::atomic<topic_t> *tokens_per_topic = new tbb::atomic<topic_t> [_num_topics]; //Storage for n(t)
	_ttc.get_counts(tokens_per_topic);

    topicCounts current_topic_counts(_num_topics); //Storage for topic counts for the current word
	int *dense_current_topic_counts = alloc<int>(_num_topics);

	// Initialize f(k) = exp(eta(k)) / (ck_k + beta_sum)
	BIT f(_num_topics);

	double *exp_eta = alloc<double>(_num_topics);
	for (int k=0; k<_num_topics; ++k)
	{
			exp_eta[k] = exp(doc->eta(k));
	}

	double exp_f_sum = 0;
	for (int k=0; k<_num_topics; ++k)
	{
			f.update(k, exp_eta[k] / (tokens_per_topic[k] + beta_sum));
			exp_f_sum += exp_eta[k] / (tokens_per_topic[k] + beta_sum);
	}

	double *c_cached_coeff = alloc<double>(_num_topics);

	for (int i=0; i<num_words_in_doc; ++i) {	//For each word
		word_t word = doc->body(i);
		topic_t old_topic = doc->topic_assignment(i);

		change_elem_t change;
		change.word = word;
		change.old_topic = old_topic;

		// Compute dense vector of current ttc
                _ttc.get_counts(word, &current_topic_counts);

		int num_tc = current_topic_counts.length;

		document_topic_counts[old_topic]--;		// Local
		tokens_per_topic[old_topic]--;			// We are just changing a copy, don't worry
		f.update(old_topic, exp_eta[old_topic] / (tokens_per_topic[old_topic] + beta_sum));

		double cwk_mass = 0;
		for (int j=0; j<current_topic_counts.length; ++j)
		{
			topic_t top = current_topic_counts.items[j].choose.top;
			cnt_t cnt = current_topic_counts.items[j].choose.cnt - (top==old_topic);

			c_cached_coeff[j] = (cwk_mass += cnt * f.values[top]);
		}

		double beta_mass = beta * f.sum;

		// Compute score
		prob[-1] = 0;

		register topic_t k;
		double sample = unif() * (cwk_mass + beta_mass);
		if (sample < cwk_mass)
		{
			k = current_topic_counts.items[num_tc-1].choose.top;
			for (int j=0; j<current_topic_counts.length; ++j)
			{
				if (c_cached_coeff[j] >= sample)
				{
					k = current_topic_counts.items[j].choose.top;
					break;
				}
			}
		}
		else
		{
			// f(k) * beta > sample 
			// f(k) > sample / beta
			sample -= cwk_mass;
			sample /= beta;

			k = f.upper_bound_sum(sample);
		}
	
		// Update changes
		topic_t new_topic = k;

		doc->set_topic_assignment(i, new_topic);
		tokens_per_topic[new_topic]++;

		f.update(k, exp_eta[k] / (tokens_per_topic[k] + beta_sum));
		document_topic_counts[new_topic]++;

		change.new_topic = k;
		if (new_topic != old_topic)
		{
				updates->push_back(change);
		}
	}
	delete[] c_cached_coeff;
	delete[] prob_buff;
	delete[] document_topic_counts;
	delete[] dense_current_topic_counts;
	delete[] tokens_per_topic;
	delete[] exp_eta;

	upd->change_list = updates;
}

void* Unigram_Model_Trainer::update(void* token) {
    update_t* upd = (update_t*) token;
    size_t sz = upd->change_list->size();
    for (size_t i = 0; i < sz; i++) {
        change_elem_t change = (*upd->change_list)[i];
        //		cout << "(" << dict->get_word(change.word) << "," << change.old_topic << "-" << change.new_topic << ")" << endl;
        //		cout << _ttc.print(change.word) << endl;
        _ttc.upd_count(change.word, change.old_topic, change.new_topic,
                ignore_old_topic);
        //		cout << _ttc.print(change.word) << endl;
    }
    if (sz != 0) {
        upd->change_list->clear();
    }
    if (upd->change_list != NULL)
        delete upd->change_list;

    return upd;
}

void* Unigram_Model_Trainer::optimize(void* token) {
    update_t* upd = (update_t*) token;
    LDA::unigram_document& doc = *(upd->doc);
    topic_t* local_topic_counts = new topic_t[_num_topics];
    topic_t* local_topic_index = new topic_t[_num_topics];
    int num_words_in_doc = doc.body_size();

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

    /************* Optimize **************/
    //		cout << "Optimizing: " << doc_index << endl;
    ++doc_index;
    if (doc_index != 0 && doc_index % tau == 0) {
        //Update global alphas with the partial
        //gradients that have been computed till now
        double* part_grads_index = part_grads;
        double *global_alphas_index = _alpha.values;
        double sum = 0.0;
        eta = pow((doc_index / tau) + 100, -0.5) * (0.1 / tau);
        for (topic_t t = 0; t < _num_topics; t++) {
            *global_alphas_index = max(
                    *global_alphas_index - eta * (*part_grads_index
                            + part_grads_top_indep), MIN_ALPHA);
            sum += *global_alphas_index;
            ++global_alphas_index;
            ++part_grads_index;
        }
        _alpha.sum = sum;

        //Reset the partial gradients
        memset(part_grads, 0, _num_topics * sizeof(double));
        part_grads_top_indep = -1 * tau * digamma(_alpha.sum);
    }
    //Update topic dependent & topic independent partial gradients
    //with contributions from this document
    ltdInd = local_topic_index;
    for (int i = 0; i < non_zero_topics; i++) {
        topic_t topic = *ltdInd;
        part_grads[topic] += digamma(_alpha.values[topic]) - digamma(
                _alpha.values[topic] + local_topic_counts[topic]);
        ++ltdInd;
    }
    part_grads_top_indep += digamma(_alpha.sum + num_words_in_doc);

    /************* Optimize **************/
    delete[] local_topic_counts;
    delete[] local_topic_index;

    return upd;
}

void* Unigram_Model_Trainer::eval(void* token, double& eval_value) {
    update_t* upd = (update_t*) token;
    LDA::unigram_document& doc = *(upd->doc);
    int num_words_in_doc = doc.body_size();

    double doc_loglikelihood = 0.0;

	double beta_sum = _beta.sum;
	double beta = _beta.values[0];

    double *theta = alloc<double>(_num_topics);
    double *eta = alloc<double>(_num_topics);
    double *phi = alloc<double>(_num_topics);
    for (topic_t topic = 0; topic < _num_topics; ++topic)
    {
	    eta[topic] = doc.eta(topic);
    }
    softmax(theta, eta, _num_topics);

    tbb::atomic<topic_t> *tokens_per_topic = new tbb::atomic<topic_t> [_num_topics]; //Storage for n(t)
    _ttc.get_counts(tokens_per_topic);

    topicCounts current_topic_counts(_num_topics); //Storage for topic counts for the current word

    for (int n=0; n<num_words_in_doc; n++)
    {
			double likelihood = 0;
	    word_t word = doc.body(n);
            _ttc.get_counts(word, &current_topic_counts);

		memset(phi, 0, sizeof(double) * _num_topics);

	    for (int j=0; j<current_topic_counts.length; ++j)
	    {
			topic_t top = current_topic_counts.items[j].choose.top;
			cnt_t cnt = current_topic_counts.items[j].choose.cnt;
			phi[top] += cnt;
		}

	    for (topic_t topic = 0; topic < _num_topics; ++topic)
	    {
		    double phi0 = (phi[topic] + beta) / (tokens_per_topic[topic] + beta_sum);
		    likelihood += theta[topic] * phi0;
	    }    

		doc_loglikelihood += log(likelihood);
    }

	delete[] theta;
	delete[] eta;
	delete[] phi;

    eval_value = doc_loglikelihood;

    return upd;
}

void* Unigram_Model_Trainer::sampleEta(void* token)
{
	update_t* upd = (update_t*) token;
	LDA::unigram_document& doc = *(upd->doc);

	// Get rng
    int rng_index = (doc.body_size() + rng_indForEta++) & RNG_MASK;
	Random* random = _randomsForEta[rng_index];

	sampleEta(doc, random);

	return upd;
}

void* Unigram_Model_Trainer::sampleEta(void* token, int thread_id)
{
	update_t* upd = (update_t*) token;
	LDA::unigram_document& doc = *(upd->doc);

	sampleEta(doc, _randomsForEta[thread_id]);

	return upd;
}

void Unigram_Model_Trainer::sampleEta(LDA::unigram_document& doc, Random *random)
{
	int num_sub_iter = Context::get_instance().get_int("subiter");
	int *ck = alloc<int>(_num_topics);
	vec eta(_num_topics);
	vec lambda(_num_topics);

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
		eta(k) = doc.eta(k);
		//lambda(k) = doc.lambda(k);
	}

	// Sample
	double *eta_mem = eta.memptr();
	double allExpSum = logExpSum(eta_mem, _num_topics);
	vec mu_col = _mu.col(0);
	for (int k=0; k<_num_topics; ++k)
	{	
		double priorVar = 1.0/_prec(k, k);	// Conditional MVGaussian
		double priorMean = 0;

		priorMean = dot(_prec.col(k), (eta - mu_col));
		priorMean -= _prec(k, k) * (eta(k) - mu_col(k));
		priorMean = mu_col(k) - priorVar*priorMean;

		for (int temp=0; temp<num_sub_iter; ++temp)
		{
			allExpSum = logExpMinus(allExpSum, eta(k));
	
			double zeta = allExpSum;
								// Swap a[k] and a[T-1] to compute logExpSum(0 ~ T-1)
	
			double rho = eta(k) - zeta;
			lambda(k) = random->nextPG(doc_length, rho);	// lambda_k:	sample from Polya-Gamma
	
			double kappa = ck[k] - (double)doc_length/2;
	
			// Compute conditional distribution with respect to lambda_k
			double tau = 1.0 / ( 1.0 / priorVar + lambda(k) );
			double gamma = tau * (priorMean/priorVar + kappa + lambda(k)*zeta);
	
			eta(k) = random->rnorm()*sqrt(tau) + gamma;//_eta_k: 	sample from normal
			allExpSum = logExpSum(allExpSum, eta(k));
		}
	}

	// Copy eta and lambda back
	for (int k=0; k<_num_topics; ++k)
	{
		doc.set_eta(k, eta(k));
		//doc.set_lambda(k, lambda(k));
	}

	delete[] ck;
}

void* Unigram_Model_Trainer::accumulateEta(void* token)
{
	update_t* upd = (update_t*) token;
	LDA::unigram_document& doc = *(upd->doc);

	accumulateEta(doc);
	return upd;
}

void Unigram_Model_Trainer::pthread_accumulateEta(
				std::vector<update_t*> &workingSet)
{
	int nsize = workingSet.size();
	_current_num_eta = nsize;

	if (nsize==0)
	{
		return;
	}

	for (int i=0; i<nsize; ++i)
		for (int k=0; k<_num_topics; ++k)
		{
			LDA::unigram_document& doc = *(workingSet[i]->doc);
			_eta_buff(k, i) = doc.eta(k);
		}

	accumulateNow();
}

void Unigram_Model_Trainer::accumulateEta(LDA::unigram_document& doc)
{
	// Stage the _eta
	for (int k=0; k<_num_topics; ++k)			
	{
		_eta_buff(k, _current_num_eta) = doc.eta(k);
	}
	++_current_num_eta;

	if (_current_num_eta == mini_batch_size)
	{
		// Use parallel Level 3 BLAS
		accumulateNow();
	}
}

void Unigram_Model_Trainer::accumulateNow()
{
	for (int i=0; i<_current_num_eta; ++i)
	{
		_eta_s += _eta_buff.col(i);
	}

	_q_s += _eta_buff.cols(0, _current_num_eta-1) 
		  * _eta_buff.cols(0, _current_num_eta-1).t();

	_num_documents += _current_num_eta;
	_current_num_eta = 0;
}

void Unigram_Model_Trainer::write(void* token) {
    update_t* upd = (update_t*) token;
    LDA::unigram_document& doc = *(upd->doc);
    doc.clear_body();

    LOG_IF(FATAL, !_tdoc_writer->write(doc) )<< "Couldn't write to the stream. Quitting..." << endl;
    delete upd;
}

void Unigram_Model_Trainer::pthread_write(void *token)
{
    update_t* upd = (update_t*) token;
    LDA::unigram_document& doc = *(upd->doc);
    doc.clear_body();

    LOG_IF(FATAL, !_tdoc_writer->write(doc) )<< "Couldn't write to the stream. Quitting..." << endl;
	delete upd->doc;
    delete upd;
}

void Unigram_Model_Trainer::sampleGauss()
{
	//accumulateNow();

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Reduce _eta_s
	wall_clock start;
	vec eta_s(_num_topics);
	double *send_eta_s = _eta_s.memptr();
	double *recv_eta_s = eta_s.memptr();
	start.tic();
	MPI_Reduce(send_eta_s, recv_eta_s, _num_topics, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	// Reduce _q_s
	mat q_s = MPI_Reduce_Symmetry_Matrix(_q_s, rank, size);

	// Compute \sum _num_documents
	int total_num_documents = 0;
	int send_num_documents = _num_documents;
	MPI_Reduce(&send_num_documents, &total_num_documents, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	sampleGauss_work(total_num_documents, rank, size, eta_s, q_s);
	
	start.tic();
	double *mu_buff = _mu.memptr();
	MPI_Bcast(mu_buff, _num_topics, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Broadcast_Symmetry_Matrix(_cov, rank, size);
	_prec = _cov.i();
}

// this._num_documents is the number of local documents
// Since we need the number of total documents across all machines,
// We use MPI_Reduce to compute the number and pass it to the actual
// sampling method.
void Unigram_Model_Trainer::sampleGauss_work(int _num_documents, int rank, int size, 
				vec &_eta_s, mat &_q_s)
{
	vec exp_mu(_num_topics);
	mat exp_cov(_num_topics, _num_topics);
	double rho;

	if (rank==0)
	{
		_eta_s /= _num_documents;

		rho = _rho.values[0];

		exp_mu = _mu_0.col(0) * (rho/(rho+_num_documents))
			   + _eta_s * (_num_documents/(rho+_num_documents));

		double factor = rho*_num_documents/(rho+_num_documents);
		exp_cov = _wishart + _q_s;
		exp_cov += (_eta_s - _mu_0.col(0)) * (_eta_s - _mu_0.col(0)).t() * factor;

		exp_cov -= _eta_s * _eta_s.t() * _num_documents;
	}
	
	_randomForGauss->NIWrnd(_mu, _cov,
					rho+_num_documents, _kappa+_num_documents, exp_mu, exp_cov,
					rank, size);
}

void Unigram_Model_Trainer::iteration_done() {
    Context& context = Context::get_instance();
    string input_w = context.get_string("input_w");
    string input_t = context.get_string("input_t");
    string output_t = context.get_string("output_t");

    string cmd = "mv " + output_t + " " + input_t;
    system(cmd.c_str());
    release_io();
    set_up_io(input_w, input_t, output_t);

	_num_documents = 0;

	_eta_s.zeros();
	_q_s.zeros();
}

void Unigram_Model_Trainer::dump(int lag)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	Context& context = Context::get_instance();
	
	ostringstream sout;
	sout << "." << lag << "." << rank;
	string dotlagdotrank = sout.str();
	
	string input_t = context.get_string("input_t");
	string output_t = context.get_string("output_t");
	
	// Copy output_t
    string input_prefix = context.get_string("inputprefix");
	string cmd = "cp " + output_t + " " + input_prefix + ".t" + dotlagdotrank;
	LOG(WARNING) << "Saving model " << cmd;
	cerr << cmd << endl;
	system(cmd.c_str()); 
	
	// dump mu
	_mu.save( input_prefix + ".mu" + dotlagdotrank );
	
	// dump cov
	_cov.save( input_prefix + ".cov" + dotlagdotrank );

	// dump ttc
	_ttc.dump( input_prefix + ".ttc" + dotlagdotrank );
}

void* Unigram_Model_Trainer::test(void* token) {
    LOG(FATAL) << "Test method called for trainer";
}

long Unigram_Model_Trainer::doc_index = -1;
