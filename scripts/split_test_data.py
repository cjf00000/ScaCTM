#!/usr/bin/python
# make a observed and heldout set
# split each document into 2 halfs

import sys, random

f_full = sys.argv[1]
f_observed = sys.argv[2]
f_heldout = sys.argv[3]

data = map(lambda x : x.replace('\r', '').replace('\n', '').split(), open(f_full).readlines())

observed = open(f_observed, 'w')
heldout = open(f_heldout, 'w')

for document in data:
	id = document[:2]
	document = document[2:]
	random.shuffle(document)

	observed_doc_len = len(document) / 2
	document_observed = document[:observed_doc_len]
	document_heldout = document[observed_doc_len:]

	observed.write(" ".join(id + document_observed) + "\n")
	heldout.write(" ".join(id + document_heldout) + "\n")

observed.close()
heldout.close()
