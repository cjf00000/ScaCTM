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
 * Merge_Dictionaries.cpp
 *
 *  Created on: 31-Jan-2011
 *      
 */

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "WordIndexDictionary.h"
#include "TopicLearner/TypeTopicCounts.h"
#include "TopicLearner/Synchronizer_Helper.h"
#include "TopicLearner/DM_Client.h"
#include <sstream>

using namespace std;

DEFINE_string(localdictprefix,"lda.dict.100", "A prefix that will be used with all files output by the program");
DEFINE_string(localttcprefix,"lda.ttc.100", "A prefix that will be used with all files output by the program");
DEFINE_string(globalttc,"lda.ttc", "A prefix that will be used with all files output by the program");
DEFINE_string(globaldictionary,"lda.dict.dump","The global dictionary; topic counts of parts of which have to be retrieved");
DEFINE_int32(topics,100,"The number of topics to be used by LDA.");
DEFINE_int32(numnodes,-1,"The number of topics to be used by LDA.");

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (strcmp(argv[i], "--help") == 0) {
            google::ShowUsageWithFlagsRestrict(argv[0], "Merge_Topic");
            exit(0);
        }
    }

    google::ParseCommandLineFlags(&argc, &argv, true);

    google::SetCommandLineOptionWithMode("minloglevel", "1",
            google::SET_FLAG_IF_DEFAULT);
    google::SetCommandLineOptionWithMode("stderrthreshold", "1",
            google::SET_FLAG_IF_DEFAULT);

    const char* pwd = google::StringFromEnv("PWD", "/tmp");
    google::SetCommandLineOptionWithMode("log_dir", pwd,
            google::SET_FLAG_IF_DEFAULT);

    LOG(WARNING)
            << "----------------------------------------------------------------------";
    LOG(WARNING) << "Log files are being stored at " << pwd
            << "/formatter.*";
    LOG(WARNING)
            << "----------------------------------------------------------------------";

    string flagsInp = google::CommandlineFlagsIntoString();

    LOG(INFO) << flagsInp << endl;

    // Global dict
    LOG_IF(FATAL,google::GetCommandLineFlagInfoOrDie("globaldictionary").is_default)
    << "You need to specify the global dictionary to be used for retrieving the global topic counts";

    WordIndexDictionary* global_dict = new WordIndexDictionary();
    string global_dict_dump = FLAGS_globaldictionary;

    LOG(WARNING) << "Initializing global dictionary from " << global_dict_dump;
    global_dict->initialize_from_dump(global_dict_dump);
    LOG(WARNING) << "global dictionary Initialized";

    int num_words = global_dict->get_num_words();
    TypeTopicCounts* ttc = new TypeTopicCounts(num_words, FLAGS_topics);

	string local_dict_prefix = FLAGS_localdictprefix;
	string local_ttc_prefix = FLAGS_localttcprefix;

	for (int node=0; node<FLAGS_numnodes; ++node)
	{
		ostringstream sout;
		sout << "." << node;
		string dotrank = sout.str();

		string local_dict_dump = local_dict_prefix + dotrank;
		string local_ttc_dump = local_ttc_prefix + dotrank;

		WordIndexDictionary* local_dict = new WordIndexDictionary();
		local_dict->initialize_from_dump(local_dict_dump);
		int local_num_words = local_dict->get_num_words();
		LOG(WARNING) << "Initialized local dump from " << local_dict_dump << " with " << local_num_words << " words.";
		local_dict->print();

		TypeTopicCounts local_ttc(local_num_words, FLAGS_topics);
		local_ttc.initialize_from_dump(local_ttc_dump, local_dict);
		LOG(WARNING) << "Initialized local ttc from " << local_ttc_dump;
		local_ttc.print();

		for (int w=0; w<local_num_words; ++w)
		{
			topicCounts tc(FLAGS_topics);

			local_ttc.get_counts(w, &tc);
			mapped_vec delta;
		    tc.convertTo(delta);

			ttc->upd_count(w, delta);
		}

		delete local_dict;
	}

	LOG(WARNING) << "Printing dict";
    global_dict->print();
	LOG(WARNING) << "Printing ttc";
    ttc->print();

    string ttc_dump = FLAGS_globalttc;
    LOG(WARNING) << "Saving it to " << ttc_dump;
    ttc->dump(ttc_dump);

    delete ttc;
    delete global_dict;
}
