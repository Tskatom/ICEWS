#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import nltk
from collections import Counter, defaultdict
import json

def get_words_collocation(event_files, num, top_k):
    gram_freq = defaultdict(int)
    for f in event_files:
        with open(f) as df:
            for line in df:
                post = json.loads(line)
                text = post["FullText"]
                tokens = nltk.wordpunct_tokenize(text)
                n_grams = nltk.collocations.ngrams(tokens, num)
                for item in n_grams:
                    gram_freq[item] += 1
        print "Finished: %s\n" % f
    
    count = sorted(gram_freq.items(), key=lambda x:x[1], reverse=True)
    return count[:top_k]

def get_count():
    count_file = "/home/tskatom/workspace/icews_model/data/handbook_count.txt"
    files = []
    with open(count_file) as cf:
        for line in cf:
            count, name = line.strip().split()
            full_name = os.path.join("/raid/tskatom/arabia_hanbook_doc_match", name)
            files.append(full_name)

    top_count = get_words_collocation(files, 2, 1000)
    out_file = open("../data/top_keywords.txt", "w")
    for item, count in top_count:
        ara_str = " ".join(item).encode('utf-8')
        out_file.write("%s\t%s\n" % (ara_str, count))
    out_file.flush()
    out_file.close()

if __name__ == "__main__":
    pass

