#include "WordIndexDictionary.h"
#include "Unigram_Model/TopicLearner/TypeTopicCounts.h"
#include "Unigram_Model/TopicLearner/TopicCounts.h"
#include "document.pb.h"
#include "DocumentReader.h"
#include <cstdio>
#include <cstring>
#include <string>
#include "matrixIO.h"
#include "types.h"
#include "utils.h"
#include "commons/TopicLearner/Parameter.h"
#include <fstream>
#include <armadillo>
using namespace arma;
using namespace std;

void dumpJson(string fileName, arma::mat &m)
{
	ofstream fout(fileName.c_str());
	if (m.n_cols == 1) {
		// Vector
		fout << "[";
		for (int k=0; k<m.n_rows; ++k)
		{
			fout << m(k, 0);

			if (k<m.n_rows-1)
				fout << ", ";
		}
		fout << "]";
	} else {
		// Matrix
		fout << "[";
		for (int x=0; x<m.n_rows; ++x)
		{
			fout << "[";
	
			for (int y=0; y<m.n_cols; ++y)
			{
				fout << m(x, y);
	
				if (y<m.n_cols-1)
					fout << ", ";
			}
	
			fout << "]";
			if (x<m.n_rows-1)
				fout << ", ";
		}
		fout << "]";
	}
	fout.close();
}

int main(int argc, char **argv)
{
	if (argc != 10)
	{
		printf("Usage: <topics> <dictionary> <ttc> <document dump prefix> <topic dump prefix> <beta> <num> <mu dump name> <cov dump name>\n");
		return 0;
	}

	int num_topics				= atoi(argv[1]);
	string dictionary_path		= string(argv[2]);
	string ttc_path				= string(argv[3]);
	string document_file_prefix	= string(argv[4]);
	string topic_file_prefix	= string(argv[5]);
	double beta					= atof(argv[6]);
	int num						= atoi(argv[7]);

        string mu_file			= string(argv[8]);
	string cov_file		= string(argv[9]);

	for (int i=0; i<argc; ++i)
	{
		puts(argv[i]);
	}

	// Read dictionary
	WordIndexDictionary dict;
	dict.initialize_from_dump(dictionary_path);
	int num_words = dict.get_num_words();

	ofstream fdict((dictionary_path + ".json").c_str());
	fdict << "[";
	for (int w=0; w<num_words; ++w)
	{
		fdict << "\"" << dict.get_word(w) << "\"";
		if (w < num_words - 1)
			fdict << ", ";
	}
	fdict << "]" << endl;
	fdict.close();

	TypeTopicCounts ttc(num_words, num_topics);
	ttc.initialize_from_dump(ttc_path, &dict);

	cerr << num_topics << " Topics." << endl;
	cerr << num_words << " Words." << endl;
	cerr << "Beta = " << beta << endl;

	ofstream fttc((ttc_path + ".json").c_str());
	ofstream fphi((ttc_path + ".phi.json").c_str());
	double **phi_wk = alloc2D<double>(num_words, num_topics);
	atomic<topic_t> *tokens_per_topic = new atomic<topic_t> [num_topics]; //Storage for n(t)
	ttc.get_counts(tokens_per_topic);
	fttc << "[";
	fphi << "[";
	arma::mat phi = zeros<mat>(num_words, num_topics);
	for (int w=0; w<num_words; ++w)
	{
		topicCounts current_topic_counts(num_topics);
		ttc.get_counts(w, &current_topic_counts);

		for (int j=0; j<current_topic_counts.length; ++j)
		{
			topic_t top = current_topic_counts.items[j].choose.top;
			cnt_t cnt = current_topic_counts.items[j].choose.cnt;

			phi_wk[w][top] += cnt;
		}

		fttc << "[";
		for (int k=0; k<num_topics; ++k)
		{
			//phi_wk[w][k] = (phi_wk[w][k] + beta) / (tokens_per_topic[k] + beta * num_words);

			fttc << phi_wk[w][k];
			if (k < num_topics-1)
				fttc << ", ";
		}
		fttc << "]" << '\n';

		if (w<num_words-1)
			fttc << ", ";

		fphi << "[";
		for (int k=0; k<num_topics; ++k)
		{
			phi_wk[w][k] = (phi_wk[w][k] + beta) / (tokens_per_topic[k] + beta * num_words);
			phi(w, k) = phi_wk[w][k];

			fphi << phi_wk[w][k];
			if (k < num_topics-1)
				fphi << ", ";
		}
		fphi << "]" << '\n';

		if (w<num_words-1)
			fphi << ", ";
	}
	fttc << "]" << endl;
	fphi << "]" << endl;
	fttc.close();
	fphi.close();
	
	// Read corpus & topic assignment
	double *theta = new double[num_topics];
	double *eta = new double[num_topics];

	ofstream ftheta((topic_file_prefix + ".json").c_str());	
	ftheta << "{";
	unigram_document wdoc, tdoc;
	int current_doc = 0;
	for (int current_topic_file=0; current_topic_file<num; ++current_topic_file)
	{
		ostringstream sout;
		sout << "." << current_topic_file;

		DocumentReader *wrdr = new DocumentReader(document_file_prefix + sout.str());
		DocumentReader *trdr = new DocumentReader(topic_file_prefix + sout.str());

		while (wrdr->read(&wdoc) != -1)
		{
			if (current_doc != 0)
			{
				ftheta << ",";				
			}
			current_doc ++;

			ftheta << "\"" << wdoc.docid() << "\" : [";
			double doc_likelihood = 0;

			trdr->read(&tdoc);
	
			int num_words_in_document = wdoc.body_size();

			for (int k=0; k<num_topics; ++k)
			{
				ftheta << tdoc.eta(k);
				if (k<num_topics-1)
					ftheta << ", ";
			}
			ftheta << "]\n";

		}
	}
	ftheta << "}";
	ftheta.close();

	arma::mat mu;
	cerr << "Mu file: " << mu_file << endl;
	mu.load(mu_file);

	arma::mat cov;
	cov.load(cov_file);

	arma::mat cosine_sim(num_topics, num_topics);
	for (int i=0; i<num_topics; ++i)
		for (int j=0; j<num_topics; ++j)
			cosine_sim(i, j) = dot(phi.col(i), phi.col(j)) / norm(phi.col(i), 2) / norm(phi.col(j), 2);

	mat corr(num_topics, num_topics);
	for (int x=0; x<num_topics; ++x)
	{
		for (int y=0; y<num_topics; ++y)
		{
			corr(x, y) = cov(x, y) / sqrt( cov(x, x) * cov(y, y) );
		}
	}

	dumpJson(mu_file + ".json", mu);
	dumpJson(cov_file + ".json", cov);
	dumpJson(cov_file + ".corr.json", corr);
	dumpJson(cov_file + ".cosine.json", cosine_sim);
}
