#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

from disco.core import Job, result_iterator
import pickle


def map(line, params):
    import json
    from nltk.corpus import stopwords
    import textblob
    badwords = stopwords.words('english')
    event_types = params["event_types"]
    event = json.loads(line)
    event_text = event["Event Sentence"]
    text_blob = textblob.TextBlob(event_text)
    pos_tags = text_blob.tags
    v_word = [t[0].lower() for t in pos_tags if t[1][:2] == 'VB']
    n_word = [t[0].lower() for t in pos_tags if t[1][:2] == 'NN']
    # lematize the words
    v_word = [textblob.Word(w).lemmatize('v') for w in v_word]
    n_word = [textblob.Word(w).lemmatize() for w in n_word]
    event_root = event["Root Code"]
    # each word only work once in each document
    good_words = set([w for w in (v_word + n_word) if w not in badwords])
    for c in event_types:
        rank_counts = {"N0": {"value": 0, "terms": {}},
                       "N1": {"value": 0, "terms": {}}}
        if c == event_root:  # the postive class
            rank_counts["N1"]["value"] = 1
            for w in good_words:
                rank_counts["N1"]["terms"][w] = 1
        else:
            rank_counts["N0"]["value"] = 1
            for w in good_words:
                rank_counts["N0"]["terms"][w] = 1
        yield c, rank_counts


def reduce(iter, params):
    from disco.util import kvgroup
    for c, infos in kvgroup(sorted(iter)):
        summary = {"N0": {"value": 0, "terms": {}},
                   "N1": {"value": 0, "terms": {}}}
        for item in infos:
            for nk, values in item.items():
                summary[nk]["value"] += values["value"]
                for term, count in values["terms"].items():
                    summary[nk]["terms"].setdefault(term, 0)
                    summary[nk]["terms"][term] += count
        yield c, summary


if __name__ == "__main__":
    import getpass
    params = {"event_types": ["14", "17", "18"]}
    f = "/home/tskatom/workspace/icews_data_labeled/" \
        "events.20141208093602.Release217.csv"
    inputs = [f]
    jobname = getpass.getuser() + ":" + "keywords"
    job = Job(jobname).run(input=inputs,
                           map=map,
                           reduce=reduce,
                           params=params)
    results = {}
    for c, infos in result_iterator(job.wait(show=True)):
        results[c] = infos

    outfile = "/home/tskatom/workspace/icews_model/data/rank_counts_mp"
    with open(outfile, 'w') as outf:
        pickle.dump(results, outf)
