#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Extract the TOP Verb in the Events
"""
__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import time
import os
import argparse
import textblob
from collections import defaultdict
from nltk.corpus import stopwords
import logging
import json
import hashlib
import pickle
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# create a file handler
handler = logging.FileHandler('%s.log' % "icews_preprocess")
handler.setLevel(logging.DEBUG)
format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(format_str)
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info('Starting Log')

COUNTRY_LIST = ["Argentina", "Brazil", "Chile", "Colombia", "Ecuador",
                "El Salvador", "Mexico", "Paraguay", "Uruguay", "Venezuela",
                "Iraq", "Egypt", "Libya", "Jordan", "Bahrain",
                "Syria", "Saudi Arabia"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--folder', type=str,
                    help='the icews input folder')
    ap.add_argument('-n', '--top', type=int, default=20,
                    help=' the top K used to limit the return keywords')
    ap.add_argument('-t', '--task', type=str,
                    help='The task name')
    ap.add_argument('-o', '--outfolder', type=str,
                    default="../data/keywords_selection",
                    help='The output folder')
    ap.add_argument('-fs', '--files', nargs='+', type=str,
                    help='the list of files to process')
    default_cameo = '/home/tskatom/workspace/icews_model/src/cameo.json'
    ap.add_argument('-c', '--cameo', type=str, default=default_cameo,
                    help='cameo mapping json')
    return ap.parse_args()


def process_keywords(args):
    """
    Summary the terms count for Feature selection and store the data into pickle
    """
    start = time.time()
    logger.info("Start process keywords %d" % start)
    folder = args.folder
    badwords = stopwords.words('english')
    event_types = ["14", "17", "18"]
    # Initiate rank_counts used to rank
    rank_counts = {}
    for c in event_types:
        rank_counts[c] = {"N1": {"value": 0, "terms": defaultdict(int)},
                          "N0": {"value": 0, "terms": defaultdict(int)}}

    for f in os.listdir(folder):
        f_start = time.time()
        logger.info('Pricessing %s at %d' % (f, f_start))
        f_f = os.path.join(folder, f)
        with open(f_f) as ff:
            for line in ff:
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
                good_words = set([w for w in
                                  (v_word + n_word) if w not in badwords])
                for c in event_types:
                    if c == event_root:  # the postive class
                        rank_counts[c]["N1"]["value"] += 1
                        for w in good_words:
                            rank_counts[c]["N1"]["terms"][w] += 1
                    else:
                        rank_counts[c]["N0"]["value"] += 1
                        for w in good_words:
                            rank_counts[c]["N0"]["terms"][w] += 1
        f_end = time.time()
        logger.info("End process %s at %d using %d" % (f,
                                                       f_end, f_end - f_start))

    end = time.time()
    logger.info("End processing keywords %d, using %d" % (end, end - start))
    outfile = os.path.join(args.outfolder, "rank_counts")
    logger.info("Dumping result to file %s" % outfile)
    with open(outfile, "w") as outf:
        pickle.dump(rank_counts, outf)

    return rank_counts


def mutual_info(rank_counts):
    """
    Rank the terms according to mutual inforation
    """
    logger.info("Start to compute Mutial Information")
    rank_results = {}
    for c in rank_counts:
        terms = set(rank_counts[c]["N1"]["terms"].keys())
        rank_results[c] = []
        n0 = 1.0 * rank_counts[c]["N0"]["value"]
        n1 = 1.0 * rank_counts[c]["N1"]["value"]
        n = n0 + n1
        for t in terms:
            mui = 0.0
            n11 = 1.0 * rank_counts[c]["N1"]["terms"].get(t, 0)
            n10 = n1 - n11
            n01 = 1.0 * rank_counts[c]["N0"]["terms"].get(t, 0)
            n00 = n0 - n01
            t_n1 = n11 + n01
            t_n0 = n10 + n00
            mui += ((n11 + 1) / (n + 4)) * np.log2(((n + 2) *
                                                    (n + 2) *
                                                    (n11 + 1))
                                                   /
                                                   ((n1 + 1) *
                                                    (t_n1 + 1) *
                                                    (n + 4)))
            mui += ((n00 + 1) / (n + 4)) * np.log2(((n + 2) *
                                                    (n + 2) *
                                                    (n00 + 1))
                                                   /
                                                   ((n0 + 1) *
                                                    (t_n0 + 1) *
                                                    (n + 4)))
            mui += ((n10 + 1) / (n + 4)) * np.log2(((n + 2) *
                                                    (n + 2) *
                                                    (n10 + 1))
                                                   /
                                                   ((n1 + 1) *
                                                    (t_n0 + 1) *
                                                    (n + 4)))
            mui += ((n01 + 1) / (n + 4)) * np.log2(((n + 2) *
                                                    (n + 2) *
                                                    (n01 + 1))
                                                   /
                                                   ((n0 + 1) *
                                                    (t_n1 + 1) *
                                                    (n + 4)))

            rank_results[c].append((t, mui))
        rank_results[c] = sorted(rank_results[c],
                                 key=lambda x: x[1], reverse=True)
    return rank_results


def ksqure_test(rank_counts):
    """
    Rank the terms according to ksquare test
    """
    logger.info("Start to compute K-square")
    rank_results = {}
    for c in rank_counts:
        terms = set(rank_counts[c]["N1"]["terms"].keys())
        rank_results[c] = []
        n0 = 1.0 * rank_counts[c]["N0"]["value"]
        n1 = 1.0 * rank_counts[c]["N1"]["value"]
        for t in terms:
            n11 = 1.0 * rank_counts[c]["N1"]["terms"].get(t, 0)
            n10 = n1 - n11
            n01 = 1.0 * rank_counts[c]["N0"]["terms"].get(t, 0)
            n00 = n0 - n01

            ksquare = (((n0 + n1) * np.power(n11 * n00 - n10 * n01, 2)) /
                       (n11 + n01) * (n11 + n10) * (n00 + n10) * (n00 + n10))
            rank_results[c].append((t, ksquare))
        rank_results[c] = sorted(rank_results[c],
                                 key=lambda x: x[1], reverse=True)
    return rank_results


def extract_feature_task(args, algo=mutual_info):
    top = args.top
    rank_file = os.path.join(args.outfolder, "rank_counts")
    if os.path.exists(rank_file):
        logger.info("Read the rank file from archive")
        rank_counts = pickle.load(open(rank_file))
    else:
        logger.info("Compute the rank counts from scratch")
        rank_counts = process_keywords(args)
    try:
        rank_words = algo(rank_counts)
    except Exception:
        logger.error("Rank Error ", exc_info=True)
    else:
        result_file = os.path.join(args.outfolder,
                                   "keywords_%s.json" % algo.__name__)
        with open(result_file, 'w') as rf:
            json.dump(rank_words, rf)
        return {k: v[:top] for k, v in rank_words.items()}


def label_icews(icews_file, outfile, map_dict):
    """
    Label the icews record with CAMEO code and assign embersId
    Parameters:
        icews_file: <str> the input icews file
        outfile: <str> the output file
    """
    not_match = set()
    with open(icews_file) as icf, open(outfile, 'w') as outf:
        try:
            keys = icf.readline().strip().split('\t')
            for line in icf:
                data = line.strip().split('\t')
                data_obj = {keys[i]: data[i] for i in range(len(data))}
                event_text = data_obj['Event Text'].lower().replace('"', "")
                if event_text not in map_dict:
                    logger.debug('Not Matching [%s] in %s' % (event_text,
                                                              icews_file))
                    not_match.add(event_text)
                    continue
                else:
                    code = map_dict[event_text]
                    data_obj["Event Code"] = code
                    data_obj["Root Code"] = code[:2]
                    data_obj["embersId"] = hashlib.sha1(
                        str(data_obj)).hexdigest()
                    outf.write(json.dumps(data_obj) + "\n")

        except Exception:
            logger.warn('Process file %s error' % icews_file, exc_info=True)
    logger.debug(not_match)
    logger.info("Finish file %s" % icews_file)


def process_label_task(args):
    folder = args.folder
    outfolder = args.outfolder
    map_dict = json.load(open(args.cameo))
    if folder:
        for f in os.listdir(folder):
            ff = os.path.join(folder, f)
            outfile = os.path.join(outfolder,
                                   f.replace("csv", "json"))
            label_icews(ff, outfile, map_dict)
    elif args.files:
        for f in args.files:
            basename = os.path.basename(f)
            basename = basename.replace("csv", "json")
            outfile = os.path.join(outfolder, basename)
            label_icews(f, outfile, map_dict)


def main():
    """
    Usage:
        python icews_keywords.py -t feature \
        -f /home/tskatom/workspace/icews_data_labeled \
        -o ../data/keywords_selection
    """
    args = parse_args()
    task = args.task
    if args.outfolder and not os.path.exists(args.outfolder):
        os.mkdir(args.outfolder)

    if task == "label":
        process_label_task(args)

    if task == "keywords":
        process_keywords(args)

    if task == "feature":
        extract_feature_task(args, algo=mutual_info)
        extract_feature_task(args, algo=ksqure_test)


if __name__ == "__main__":
    main()
