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
using namespace std;

int main(int argc, char **argv)
{
	if (argc != 8)
	{
		printf("Usage: <topics> <dictionary> <ttc> <document dump prefix> <topic dump prefix> <beta> <num>\n");
		return 0;
	}

	int num_topics				= atoi(argv[1]);
	string dictionary_path		= string(argv[2]);
	string ttc_path				= string(argv[3]);
	string document_file_prefix	= string(argv[4]);
	string topic_file_prefix	= string(argv[5]);
	double beta					= atof(argv[6]);
	int num						= atoi(argv[7]);

	// Read dictionary
	WordIndexDictionary dict;
	dict.initialize_from_dump(dictionary_path);
//	dict.print();
	int num_words = dict.get_num_words();

	TypeTopicCounts ttc(num_words, num_topics);
	ttc.initialize_from_dump(ttc_path, &dict);
//	ttc.print();

	cerr << num_topics << " Topics." << endl;
	cerr << num_words << " Words." << endl;
	cerr << "Beta = " << beta << endl;

	double **phi_wk = alloc2D<double>(num_words, num_topics);
	atomic<topic_t> *tokens_per_topic = new atomic<topic_t> [num_topics]; //Storage for n(t)
	ttc.get_counts(tokens_per_topic);
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

		for (int k=0; k<num_topics; ++k)
		{
			phi_wk[w][k] = (phi_wk[w][k] + beta) / (tokens_per_topic[k] + beta * num_words);
		}
	}
//	print(phi_wk, num_words, num_topics);
	
	// Read corpus & topic assignment
	double total_log_likelihood = 0;
	int total_words = 0;

	double *theta = new double[num_topics];
	double *eta = new double[num_topics];

	unigram_document wdoc, tdoc;
	
	for (int current_topic_file=0; current_topic_file<num; ++current_topic_file)
	{
		ostringstream sout;
		sout << "." << current_topic_file;

		DocumentReader *wrdr = new DocumentReader(document_file_prefix + sout.str());
		DocumentReader *trdr = new DocumentReader(topic_file_prefix + sout.str());

		while (wrdr->read(&wdoc) != -1)
		{
			double doc_likelihood = 0;

			trdr->read(&tdoc);
	
			int num_words_in_document = wdoc.body_size();
			total_words += num_words_in_document;
	
			for (int k=0; k<num_topics; ++k)
			{
				eta[k] = tdoc.eta(k);
			}
	
			softmax(theta, eta, num_topics);
	
			for (int n=0; n<num_words_in_document; ++n)
			{
				word_t word = wdoc.body(n);
				double likelihood = 0;
				for (topic_t topic = 0; topic < num_topics; ++topic)
				{
					likelihood += phi_wk[word][topic] * theta[topic];
				}
				total_log_likelihood += log(likelihood);
				doc_likelihood += log(likelihood);
			}

			//printf("%lf\n", exp( -doc_likelihood / num_words_in_document ));
		}
	}

	printf("%lf\n", exp(-total_log_likelihood / total_words));
}
