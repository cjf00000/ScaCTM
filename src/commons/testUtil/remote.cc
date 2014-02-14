#include <cstdio>
#include <vector>
#include <cassert>
#include "remote.h"
#include "DocumentReader.h"
#include "DocumentWriter.h"
#include "document.pb.h"
#include <sstream>
using namespace std;

void spreadDocument(const char *prefix, int nnodes)
{
	DocumentReader reader(prefix);	

	unigram_document doc;
	vector<unigram_document> docs;

	while (reader.read(&doc) != -1)
	{
		docs.push_back(doc);
	}

	int n = docs.size();
	assert( n % nnodes == 0 );
	int p = n/nnodes;

	int cnt = 0;
	for (int i=1; i<=nnodes; ++i)
	{
		char path[256];
		sprintf(path, "%s.%d", prefix, i);
		DocumentWriter writer(path);

		for (; cnt<p*i; ++cnt)
		{
			writer.write(docs[cnt]);
		}

		char buff[256];
		sprintf(buff, "scp %s juncluster%d:/home/jianfei/git/yahoo-lda/%s", path, i, prefix);
		puts(buff);
		system(buff);
	}
}

vector<unigram_document> collectDocuments(const char *prefix, int nnodes)
{
	vector<unigram_document> docs;

	for (int i=1; i<=nnodes; ++i)
	{	
		char path[256];
		sprintf(path, "%s.%d", prefix, i);
		char buff[256];
		sprintf(buff, "scp juncluster%d:/home/jianfei/git/yahoo-lda/%s %s", i, prefix, path);
		puts(buff);
		system(buff);

		DocumentReader reader(path);
		unigram_document doc;

		while (reader.read(&doc) != -1)
		{
			docs.push_back(doc);
		}

		printf("%d\n", docs.size());
	}

	return docs;
}

void broadcastFile(const char *path, int nnodes)
{
	for (int i=1; i<=nnodes; ++i)
	{
		char buff[256];
		sprintf(buff, "scp %s juncluster%d:/home/jianfei/git/yahoo-lda/%s", path, i, path);
		puts(buff);
		system(buff);
	}
}

void removeAll(const char *path, int nnodes)
{
	for (int i=1; i<=nnodes; ++i)
	{
		char buff[256];
		sprintf(buff, "ssh juncluster%d rm /home/jianfei/git/yahoo-lda/%s", i, path);		
		puts(buff);
		system(buff);
	}
}

void startParameterServer(int nnodes)
{
	ostringstream sout;
	sout << "nohup mpiexec -f hostfile";

	for (int i=0; i<nnodes; ++i)
	{
		sout << " -n 1 ./DM_Server 1 " << i << " " << nnodes << " juncluster"
		     << i+1 << ":9876 --Ice.ThreadPool.Server.SizeMax=9 ";
		if (i<nnodes-1)
			sout << ":";	
	}
	sout << " &";
	puts(sout.str().c_str());
	system(sout.str().c_str());
}

void endParameterServer(int nnodes)
{
	for (int i=1; i<=nnodes; ++i)
	{
		char buff[256];
		sprintf(buff, "ssh juncluster%d ""killall DM_Server &"" ", i);
		puts(buff);
		system(buff);
	}
}

void mergeDict(int nnodes)
{
	for (int i=1; i<=nnodes; ++i)
	{
		char buff[256];
		sprintf(buff, "scp juncluster%d:/home/jianfei/git/yahoo-lda/lda.dict.dump lda.dict.dump.%d",
				i, i-1);
		puts(buff);
		system(buff);
	}

	char buff[256];
	sprintf(buff, "/home/jianfei/git/yahoo-lda/Merge_Dictionaries \
-dictionaries=%d -dumpprefix=lda.dict.dump", nnodes);
	puts(buff);
	system(buff);
}

void mergeTTC(int ntopics, const char* serverlist)
{
	char buff[256];
	sprintf(buff, "/home/jianfei/git/yahoo-lda/Merge_Topic_Counts -topics=%d -clientid=0 -servers=%s -globaldictionary=lda.dict.dump",
			ntopics, serverlist);
	puts(buff);
	system(buff);
}
