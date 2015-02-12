#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import nltk
from collections import defaultdict
from operator import itemgetter
import glob
import codecs
from collections import Counter

def extract_keywords(event_type, version, top_k):
    data_folder = "/raid/tskatom/icews_sentences/%s/%s/" % (version, event_type)
    files = glob.glob(data_folder + "*")
    sentences = []
    for f in files:
        with codecs.open(f, encoding='utf-8') as df:
            for line in df:
                date, sentence = line.strip().split("\t")
                sentences.append(sentence)

    print "Total [%d] Events" % len(sentences)
    
    words_candidate = defaultdict(int)
    #tokenize and get pos
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        pos = nltk.pos_tag(tokens)
        #we only interest in verb
        for w, p in pos:
            if p[:2] == "VB":
                words_candidate[w] += 1

    #sort the candidates by frequency
    sorted_keywords = sorted(words_candidate.items, key=itemgetter(1), reverse=True)
    
    print "Total[%d] keywrods" % len(sorted_keywords)
    
    return sorted_keywords[:top_k]

def get_freq_icews_keywords(event_type, top_k):
    data_file = "../data/%s.txt" % event_type
    words = []
    with open(data_file) as df:
        for line in df:
            for item in line.strip().split():
                word, tag = item.rsplit("/",1)
                if tag[:2] == "VB":
                    words.append(word.lower())
    word_freq = Counter(words)
    top_words = word_freq.most_common(top_k)
    with open('%s_most_common.txt' % event_type, 'w') as wf:
        for w, count in top_words:
            wf.write("%s\t%s\n" % (w, count))
    return top_words

if __name__ == "__main__":
    event_types = ["17", "18"]
    for et in event_types:
        get_freq_icews_keywords(et, 300)

