#ifndef __REMOTE
#define __REMOTE

#include <vector>
#include "document.pb.h"
using namespace std;

void spreadDocument(const char *prefix, int nnodes);
vector<LDA::unigram_document> collectDocuments(const char *prefix, int nnodes);
void broadcastFile(const char *path, int nnodes);
void removeAll(const char *path, int nnodes);

void startParameterServer(int nnodes);
void endParameterServer(int nnodes);

void mergeDict(int nnodes);
void mergeTTC(int ntopics, const char *serverlist);

#endif
