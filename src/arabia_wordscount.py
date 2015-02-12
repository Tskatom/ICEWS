#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import glob
import json
import pandas as pd
import argparse

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inFolder", type=str)
    ap.add_argument("--outFolder", type=str)
    args = ap.parse_args()
    return args

def keyword_count(inFolder, outFolder):
    files = glob.glob(os.path.join(inFolder, "arabia*"))

    dailyCount = {}
    for f in files:
        with open(f) as countf:
            print f
            counts = json.load(countf)
            for country, values in counts.items():
                for keyword, day_counts in values.items():
                    for day, count in day_counts.items():
                        dailyCount.setdefault(country, {})
                        dailyCount[country].setdefault(day, {})
                        dailyCount[country][day].setdefault(keyword, 0)
                        dailyCount[country][day][keyword] += count
    #output file
    for country in dailyCount:
        filename = os.path.join(outFolder, "%s_keywords_count.json" % country.replace(" ", "_"))
        with open(filename, "w") as of:
            json.dump(dailyCount[country], of)


def main():
    args = parse()
    inFolder = args.inFolder
    outFolder = args.outFolder
    dailyCount = keyword_count(inFolder, outFolder)

if __name__ == "__main__":
    sys.exit(main())

