#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

from networkx.readwrite import json_graph
import numpy as np
from textblob import TextBlob
import nltk
import re
import sys
import os
from multiprocessing import Process, Queue
import json
import argparse
import glob
import nltk.data
from dateutil import parser
from collections import Counter
import codecs
from util import logs
import networkx as nx

logs.init(l=logs.logging.DEBUG)
logger = logs.getLogger(__name__)

COUNTRY_ARABIC = {"العراق": "Iraq",
        "مصر": "Egypt",
        "ليبيا": "Libya",
        "الاردن": "Jordan",
        "البحرين": "Bahrain",
        "سوريا": "Syria",
        "المملكة العربية السعودية": "Saudi Arabia"}

MENA_CAP_DICT = {
    #"عمّان": "Jordan",
    u'\u0628\u063a\u062f\u0627\u062f': "Iraq",
    u'\u0627\u0644\u0642\u0627\u0647\u0631\u0629': "Egypt",
    u'\u0637\u0631\u0627\u0628\u0644\u0633': "Libya",
    u'\u0627\u0644\u0645\u0646\u0627\u0645\u0629': "Bahrain",
    u'\u062f\u0645\u0634\u0642': "Syria",
    u'\u0627\u0644\u0631\u064a\u0627\u0636': "Saudi Arabia"
}

COUNTRY_ENG = ["Iraq", "Egypt", "Libya", "Jordan", "Bahrain", "Syria", "Saudi Arabia"]

def worker(task_queue, result_queue=None):
    for task in iter(task_queue.get, 'STOP'):
        result = handler(task)
        if result_queue is not None:
            result_queue.put(result)

def handler(task):
    work_func = task["func"]
    result = work_func(task["params"])
    return result

def create_event_count_task(args, task_queue):
    inFolder = args.inFolder
    outFolder = args.outFolder
    countLevel = args.countLevel
    files = glob.glob(os.path.join(inFolder, "*"))
    for f in files:
        task = {}
        task["func"] = event_count
        params = {}
        params["dayFile"] = f
        params["outFolder"] = outFolder
        params["countLevel"] = countLevel
        task["params"] = params
        task_queue.put(task)
    task_count = len(files)
    return task_count

def event_count(params):
    dayFile = params["dayFile"]
    outFolder = params["outFolder"]
    basename = os.path.basename(dayFile)
    outFile = os.path.join(outFolder, basename)
    countLevel = params["countLevel"]
    e_count = {}
    with open(dayFile) as df, open(outFile, "w") as outf:
        for line in df:
            record = json.loads(line)
            event_date = parser.parse(record["DocDS"]).strftime("%Y-%m-%d")
            if countLevel == "city":
                places = record.get("Places", None)
                if type(places) != list:
                    continue
                for p in places:
                    loc = p["Name"].strip().split("-")
                    if len(loc) <= 1:
                        continue
                    else:
                        p_loc = loc[1].strip()
                        if p_loc in MENA_CAP_DICT:
                            p_loc_cou = MENA_CAP_DICT[p_loc]
                            e_count.setdefault(p_loc_cou,{})
                            e_count[p_loc_cou].setdefault(event_date, 0)
                            e_count[p_loc_cou][event_date] += 1

        json.dump(e_count, outf)

def create_eventLocGroup_task(args, task_queue):
    inFolder = args.inFolder
    outFolder = args.outFolder
    countLevel = args.countLevel
    files = glob.glob(os.path.join(inFolder, "*"))
    for f in files:
        task = {}
        task["func"] = event_groupby_location
        params = {}
        params["dayFile"] = f
        params["outFolder"] = outFolder
        params["countLevel"] = countLevel
        task["params"] = params
        task_queue.put(task)
    task_count = len(files)
    return task_count

def event_groupby_location(params):
    dayFile = params["dayFile"]
    outFolder = params["outFolder"]
    basename = os.path.basename(dayFile)
    countLevel = params["countLevel"]
    event_group = {}
    with open(dayFile) as df:
        for line in df:
            try:
                record = json.loads(line)
                event_date = parser.parse(record["DocDS"]).strftime("%Y-%m-%d")
                if countLevel == "city":
                    places = record.get("Places", None)
                    if type(places) != list:
                        continue
                    for p in places:
                        loc = p["Name"].strip().split("-")
                        if len(loc) <= 1:
                            continue
                        else:
                            p_loc = loc[1].strip()
                            if p_loc in MENA_CAP_DICT:
                                p_loc_cou = MENA_CAP_DICT[p_loc]
                                event_group.setdefault(p_loc_cou, [])
                                event_group[p_loc_cou].append(record)
            except:
                logger.info('Error:%s' % dayFile, exc_info=True)
                continue

    for country in event_group:
        final_folder = os.path.join(outFolder, country.replace(" ","_"))
        if not os.path.exists(final_folder):
            os.mkdir(final_folder)
        with open(os.path.join(final_folder, basename), 'w') as outf:
            for event in event_group[country]:
                outf.write(json.dumps(event) + '\n')



def create_keywords_count_task(args, task_queue):
    files = glob.glob(os.path.join(args.inFolder, "arabia*"))
    files.sort(reverse=False)
    traceOutFolder = args.outFolder  + "_trace"
    if not os.path.exists(traceOutFolder):
        os.mkdir(traceOutFolder)
    start = args.start
    end = args.end
    pattern = re.compile('\d{4}-\d{2}-\d{2}')
    for f in files:
        basename = os.path.basename(f)
        found = pattern.findall(basename)
        if len(found) > 0:
            baseday = found[0]
        else:
            continue
        if baseday < start or baseday > end:
            continue
        param = {}
        param["dayFile"] = f
        param["outFolder"] = args.outFolder
        param["traceOutFolder"] = traceOutFolder
        param["logFolder"] = args.logFolder
        param["keywordsFile"] = args.keywordsFile
        task = {"params": param, "func": keyword_count}
        task_queue.put(task)
    task_count = len(files)
    return task_count

def keyword_count(param):
    dayFile = param["dayFile"]
    ruleFile = param["keywordsFile"]
    outFolder = param["outFolder"]
    traceOutFolder = param["traceOutFolder"]

    basename = os.path.basename(dayFile)
    outFile = os.path.join(outFolder, basename)
    keywords = []
    with codecs.open(ruleFile, encoding='utf-8') as rf:
        keywords = [l.strip() for l in rf]

    result = {}

    trace_file = os.path.join(traceOutFolder, basename)
    #set up inverse index to store the matched keywords and doc infoa
    inverse_result = {}
    with open(dayFile) as df, open(outFile,'w') as of, open(trace_file, 'w') as tf:
        for line in df:
            try:
                post = json.loads(line)
                eid = post["embersId"]
                #fullText = post["FullText"]
                fullText = post["FullText"][:80] # we use the first 40 character to match
                post_date = parser.parse(post['date']).strftime('%Y-%m-%d')
                country = post["PublishCountryE"]
                if country not in COUNTRY_ENG:
                    contiue

                doc_words = {}
                for keyword in keywords:
                    count = fullText.count(keyword)
                    result.setdefault(country, {})
                    result[country].setdefault(keyword, {})
                    result[country][keyword].setdefault(post_date,0)
                    result[country][keyword][post_date] += count
                    inverse_result.setdefault(country, {})
                    inverse_result[country].setdefault(keyword, {})
                    inverse_result[country][keyword].setdefault(post_date, [])
                    if count > 0:
                        doc_words[keyword] = count
                        inverse_result[country][keyword][post_date].append((eid, count))
                if len(doc_words) > 0:
                    post['keywords_count'] = doc_words
                    #tf.write((json.dumps(post, ensure_ascii=False) + "\n").decode('utf-8'))
            except:
                continue
        of.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
        tf.write(json.dumps(inverse_result,ensure_ascii=False).encode('utf-8'))
    print "Done with %s\n" % dayFile

def create_document_matched_task(args, task_queue):
    files = glob.glob(os.path.join(args.inFolder, "arabia*"))
    files.sort(reverse=False)
    for f in files:
        param = {}
        param["dayFile"] = f
        param["outFolder"] = args.outFolder
        param["logFolder"] = args.logFolder
        param["keywordsFile"] = args.keywordsFile
        param["match_region"] = args.match_region
        task = {"params": param, "func": document_matched_detail}
        task_queue.put(task)
    task_count = len(files)
    return task_count

def document_matched_detail(param):
    dayFile = param["dayFile"]
    match_region = param["match_region"]
    ruleFile = param["keywordsFile"]
    outFolder = param["outFolder"]
    basename = os.path.basename(dayFile)
    outFile = os.path.join(outFolder, basename)
    keywords = []
    with codecs.open(ruleFile) as rf:
        keywords = [l.strip() for l in rf]
    rule = "|".join(keywords).decode("utf-8")
    pattern = re.compile(rule, re.U)
    with open(dayFile) as df, open(outFile, 'w') as of:
        for line in df:
            try:
                post = json.loads(line)
                if match_region == "full":
                    fullText = post["FullText"]
                elif match_region == "lead":
                    blob = TextBlob(post["FullText"])
                    # choose first 3 sentences as search range
                    sentences = [s for s in blob.raw_sentences]
                    fullText = post['Caption'] + u' ' + u" ".join(sentences[:3])
                matched = pattern.findall(fullText)
                post[u'matched'] = {k:v for k, v in Counter(matched).items()}
                #write to outfile if and only if the keys are not zero
                if len(post[u'matched']) > 0:
                    #print post
                    of.write(json.dumps(post, ensure_ascii=False).encode('utf-8') + "\n")
            except:
                continue
    print "Done with %s\n" % dayFile

def create_document_count_task(args, task_queue):
    files = glob.glob(os.path.join(args.inFolder, "arabia*"))
    files.sort(reverse=False)
    task_count = 0
    outFolder = args.outFolder
    filter_folder = outFolder
    if filter_folder[-1] == "/":
        filter_folder = filter_folder[:-1]
    filter_folder = filter_folder + "_filter"

    if not os.path.exists(filter_folder):
        os.mkdir(filter_folder)

    for f in files:
        day = re.findall('\d{4}-\d{2}-\d{2}', f)[0]
        if args.start > day or args.end < day:
            continue
        param = {}
        param["dayFile"] = f
        param["outFolder"] = args.outFolder
        param["keywordsFile"] = args.keywordsFile
        param["filterFolder"] = filter_folder
        task = {"params": param, "func": document_count}
        task_queue.put(task)
        task_count += 1
    return task_count

def document_count(param):
    dayFile = param["dayFile"]
    ruleFile = param["keywordsFile"]
    outFolder = param["outFolder"]
    basename = os.path.basename(dayFile)
    outFile = os.path.join(outFolder, basename)

    keywords = []

    filter_folder = param["filterFolder"]
    filterfile = os.path.join(filter_folder, basename)

    with open(ruleFile) as rf:
        keywords = [l.strip().decode('utf-8') for l in rf]
    rule = "|".join(keywords)
    pattern = re.compile(rule, re.U)
    result = {}
    #we try to store the filtered news when do the filtering work
    filtering_result = {}
    with open(dayFile) as df, open(outFile, 'w') as of, open(filterfile, "w") as ff:
        for line in df:
            try:
                post = json.loads(line)
                if "FullText" not in post:
                    continue
                fullText = post["FullText"][:80]
                post_date = parser.parse(post['date']).strftime("%Y-%m-%d")
                if "PublishCountryE" not in post or "PublishCountryA" not in post:
                    continue
                pub_country_e = post["PublishCountryE"]
                pub_country_a = post["PublishCountryA"]
                places = post["Places"]

                countries = []
                #set up the country list for the posts
                if places is None:
                    if pub_country_e in COUNTRY_ENG:
                        countries.append(pub_country_e)
                else:
                    places = set([p["Name"].split("-")[0].encode("utf-8").strip() for p in places])
                    for p_country in places:
                        if p_country in COUNTRY_ARABIC:
                            countries.append(COUNTRY_ARABIC[p_country])
                if len(countries) == 0:
                    continue

                matched = pattern.findall(fullText)
                if matched and len(matched) > 1:
                    for country in countries:
                        result.setdefault(country, {})
                        result[country].setdefault(post_date, 0)
                        filtering_result.setdefault(country, {})
                        filtering_result[country].setdefault(post_date, [])
                        result[country][post_date] += 1
                        filtering_result[country][post_date].append(post)
            except:
                print "Error: ", sys.exc_info()
                continue
        json.dump(result, of)
        json.dump(filtering_result, ff)

    #output the final results
    print "Done with %s\n" % dayFile

def create_top_document_count_task(args, task_queue):
    files = glob.glob(os.path.join(args.inFolder, "arabia*"))
    files.sort(reverse=False)
    task_count = 0
    outFolder = args.outFolder
    filter_folder = outFolder
    if filter_folder[-1] == "/":
        filter_folder = filter_folder[:-1]
    filter_folder = filter_folder + "_top"

    if not os.path.exists(filter_folder):
        os.mkdir(filter_folder)

    for f in files:
        day = re.findall('\d{4}-\d{2}-\d{2}', f)[0]
        if args.start > day or args.end < day:
            continue
        param = {}
        param["dayFile"] = f
        param["outFolder"] = args.outFolder
        param["keywordsFile"] = args.keywordsFile
        param["filterFolder"] = filter_folder
        task = {"params": param, "func": top_document_count}
        task_queue.put(task)
        task_count += 1
    return task_count

def top_document_count(param):
    dayFile = param["dayFile"]
    ruleFile = param["keywordsFile"]
    outFolder = param["outFolder"]
    basename = os.path.basename(dayFile)
    outFile = os.path.join(outFolder, basename)

    keywords = []

    filter_folder = param["filterFolder"]
    filterfile = os.path.join(filter_folder, basename)

    with open(ruleFile) as rf:
        keywords = [l.strip().decode('utf-8') for l in rf]
    rule = "|".join(keywords)
    pattern = re.compile(rule, re.U)
    result = {}
    #we try to store the filtered news when do the filtering work
    filtering_result = {}
    with open(dayFile) as df, open(filterfile, "w") as ff:
        for line in df:
            try:
                post = json.loads(line)
                if "FullText" not in post:
                    continue
                fullText = post["FullText"]
                post_date = parser.parse(post['date']).strftime("%Y-%m-%d")
                if "PublishCountryE" not in post or "PublishCountryA" not in post:
                    continue
                pub_country_e = post["PublishCountryE"]
                pub_country_a = post["PublishCountryA"]
                places = post["Places"]

                countries = []
                #set up the country list for the posts
                if places is None:
                    if pub_country_a in COUNTRY_ARABIC:
                        countries.append(pub_country_e)
                else:
                    places = set([p["Name"].split("-")[0].encode("utf-8").strip() for p in places])
                    for p_country in places:
                        if p_country in COUNTRY_ARABIC:
                            countries.append(COUNTRY_ARABIC[p_country])
                if len(countries) == 0:
                    continue

                matched = pattern.findall(fullText)
                if matched and len(matched) > 1:
                    logger.debug("Matched %s" % json.dumps(matched))
                    for country in countries:
                        result.setdefault(country, {})
                        result[country].setdefault(post_date, 0)
                        filtering_result.setdefault(country, {})
                        filtering_result[country].setdefault(post_date, [])
                        result[country][post_date] += 1
                        filtering_result[country][post_date].append(post)
            except:
                print "Error: ", sys.exc_info()
                logger.error("First step match", exc_info=True)
                continue
        json.dump(filtering_result, ff)

    #Second step, compute frequency of the top n-grams and the volume of posts containing the terms
    country_top_words = {}
    for country in filtering_result:
        country_top_words[country] = []
        n_grams = []
        for day in filtering_result[country]:
            for post in filtering_result[country][day]:
                caption = post['Caption']
                fullText = post['FullText']
                full_text_blob = TextBlob(fullText)
                lead_sent = full_text_blob.sentences[0]
                caption_blob = TextBlob(caption)
                n_grams += caption_blob.ngrams(n=2)
                n_grams += lead_sent.ngrams(n=2)

        hash_n_grams = [u' '.join(wl) for wl in n_grams]
        c = Counter(hash_n_grams)
        top_freqs = []
        k = 20
        for w in c.most_common(k*2):
            if k <= 0:
                break
            try:
                w[0].encode('ascii')
            except UnicodeEncodeError:
                #means Arabia languge
                top_freqs.append((w[0], w[1]))
                k -= 1
        country_top_words[country] = top_freqs

    #construct country search keywords
    country_rule = {}
    for country in country_top_words:
        rule = []
        for w in country_top_words[country]:
            try:
                temp_p = re.compile(w[0], re.U)
            except:
                logger.warn("Invalid re expression %s" % w[0])
            else:
                rule.append(u"%s" % w[0])
        rule = u'|'.join(rule)
        country_rule[country] = rule
    #find the frequency of the keywords in daily post files
    final_doc_results = {}
    final_top_results = {}
    with open(dayFile) as df, open(outFile, "w") as of:
        for line in df:
            try:
                post = json.loads(line)
                if "FullText" not in post:
                    continue
                fullText = post["FullText"]
                caption = post['Caption']
                full_text_blob = TextBlob(fullText)
                lead_sent = full_text_blob.sentences[0]
                post_date = parser.parse(post['date']).strftime("%Y-%m-%d")
                if "PublishCountryE" not in post or "PublishCountryA" not in post:
                    continue
                pub_country_e = post["PublishCountryE"]
                pub_country_a = post["PublishCountryA"]
                places = post["Places"]

                countries = []
                #set up the country list for the posts
                if places is None:
                    if pub_country_a in COUNTRY_ARABIC:
                        countries.append(pub_country_e)
                else:
                    places = set([p["Name"].split("-")[0].encode("utf-8").strip() for p in places])
                    for p_country in places:
                        if p_country in COUNTRY_ARABIC:
                            countries.append(COUNTRY_ARABIC[p_country])
                if len(countries) == 0:
                    continue

                text = caption + u' ' + lead_sent.string
                logger.debug("Countries %s" % json.dumps(countries))
                for country in countries:
                    rule = country_rule.get(country, None)
                    if rule is None:
                        continue
                    pattern = re.compile(rule, re.U)
                    matched = pattern.findall(text)
                    if matched and len(matched) > 1:
                        final_doc_results.setdefault(country, {})
                        final_doc_results[country].setdefault(post_date, {"count":0, "postIds":[],"terms":{}, "titles":[]})
                        final_doc_results[country][post_date]["count"] += 1
                        final_doc_results[country][post_date]["postIds"].append(post["embersId"])
                        final_doc_results[country][post_date]["titles"].append(text)
                        for term in matched:
                            final_doc_results[country][post_date]["terms"].setdefault(term, 0)
                            final_doc_results[country][post_date]["terms"][term] += 1
            except:
                print "Error: ", sys.exc_info()
                logger.error("Error second step map!" , exc_info=True)
                continue
        json.dump(final_doc_results, of)

    print "Done with %s\n" % dayFile

def pos_sens(param):
    filename = param["filename"]
    outFolder = param["outFolder"]
    basename = os.path.basename(filename)
    outfile = os.path.join(outFolder, basename)
    with open(filename) as f, open(outfile, 'w') as outf:
        for l in f:
            day, doc = l.strip().decode('utf-8').split('\t')
            sens = nltk.sent_tokenize(doc)
            sens = [nltk.word_tokenize(s) for s in sens]
            sens = [nltk.pos_tag(s) for s in sens]
            sens_str = json.dumps(sens)
            outf.write(day.decode('utf-8') + u'\t' + sens_str + u"\n")
    print "Done with %s." % filename


def create_pos_sens_task(args, task_queue):
    infolder = args.inFolder
    outfolder = args.outFolder
    files = glob.glob(os.path.join(infolder, "*"))
    task_count = 0
    for f in files:
        param = {}
        param["filename"] = f
        param["outFolder"] = outfolder
        task = {"params": param, "func": pos_sens}
        task_queue.put(task)
        task_count += 1
    return task_count

def crete_event_group_task(args, task_queue):
    infolder = args.inFolder
    outfolder = args.outFolder
    files = glob.glob(os.path.join(infolder, "*"))
    task_count = 0
    for f in files:
        param = {}
        param["filename"] = f
        param["outFolder"] = outfolder
        task = {"params": param, "func": event_group}
        task_queue.put(task)
        task_count += 1
    return task_count

def event_group(param):
    filename = param["filename"]
    outFolder = param["outFolder"]
    outfile = os.path.join(outFolder, os.path.basename(filename))
    with open(filename) as df, open(outfile, 'w') as dp:
        country_posts = {}
        for l in df:
            post = json.loads(l)
            places = post["Places"]
            countries = []
            #set up the country list for the posts
            if places is None:
                pass
            else:
                places = set([p["Name"].split("-")[0].encode("utf-8").strip() for p in places])
                for p_country in places:
                    if p_country in COUNTRY_ARABIC:
                        countries.append(COUNTRY_ARABIC[p_country])
            if len(countries) == 0:
                continue
            for country in countries:
                country_posts.setdefault(country, [])
                country_posts[country].append(post)

        for country in country_posts:
            g = nx.Graph()
            posts = country_posts[country]
            count = len(posts)
            sim_matrix = []
            for i in range(count):
                p_i = posts[i]
                g.add_node(p_i['embersId'], cap=p_i['Caption'])
                for j in range(i+1, count):
                    p_j = posts[j]
                    try:
                        score = similarity(p_i, p_j)
                        if score >= 0.75:
                            sim_matrix.append((p_i['embersId'], p_j['embersId'], score))
                    except:
                        print sys.exc_info()
                        continue

            g.add_weighted_edges_from(sim_matrix)
            conn_events = sorted(nx.connected_components(g), key = len, reverse=True)
            g_data = json_graph.node_link_data(g)
            g_data['country'] = country
            g_data_str = json.dumps(g_data)
            dp.write(g_data_str + u"\n")

def similarity(p1, p2):
    text1 = p1['Caption']
    token1 = Counter(text1.split())
    text2 = p2['Caption']
    token2 = Counter(text2.split())
    words = list(set(token1.keys() + token2.keys()))
    v1 = np.zeros(len(words))
    v2 = np.zeros(len(words))
    for i in range(len(words)):
        v1[i] = token1[words[i]]
        v2[i] = token2[words[i]]
    score = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return score

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inFolder', type=str)
    ap.add_argument("--outFolder", type=str)
    ap.add_argument("--logFolder", type=str)
    ap.add_argument("--headerFile", type=str)
    ap.add_argument("--keywordsFile", type=str)
    ap.add_argument('--core', type=int)
    ap.add_argument('--task', type=str)
    ap.add_argument('--start', type=str)
    ap.add_argument('--end', type=str)
    ap.add_argument('--code', type=str)
    ap.add_argument('--countLevel', type=str)
    ap.add_argument('--match_region', type=str, default='full')
    return ap.parse_args()


def main():
    args = parse_args()
    task_queue = Queue()
    result_queue = None

    if args.outFolder and not os.path.exists(args.outFolder):
        os.mkdir(args.outFolder)

    if args.logFolder and not os.path.exists(args.logFolder):
        os.system("mkdir -p %s" % args.logFolder)


    if args.task == "keywordsCount":
        task_count = create_keywords_count_task(args, task_queue)
    elif args.task == "documentCount":
        task_count = create_document_count_task(args, task_queue)
    elif args.task == "documentMatch":
        task_count = create_document_matched_task(args, task_queue)
    elif args.task == "possens":
        task_count = create_pos_sens_task(args, task_queue)
    elif args.task == "topDocumentCount":
        task_count = create_top_document_count_task(args, task_queue)
    elif args.task == "eventGroup":
        task_count = crete_event_group_task(args, task_queue)
    elif args.task == "eventCount":
        task_count = create_event_count_task(args, task_queue)
    elif args.task == 'eventLocGroup':
        task_count = create_eventLocGroup_task(args, task_queue)

    for i in range(args.core):
        Process(target=worker, args=(task_queue,result_queue)).start()
        task_queue.put('STOP')


    print "------------------I am Done"


if __name__ == "__main__":
    main()
