#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import argparse
import json
import glob
from datetime import datetime
from util import logs

logs.init()
logger = logs.getLogger(__name__)


class Arabia:
    def e_related_docs(self, files, storedfile=None):
        """
        Extract the related docs count from files
        input:
            files: <list> the list of files which are
                the output of top_document_count task
            storedfile: <str> the filepath to store the extraced features
        return:
            results: <dict> the daily related event counts
        """
        results = {}
        for f in files:
            logger.info("Start to process %s" % f)
            with open(f) as df:
                daily_info = json.load(df)
                for country in daily_info:
                    results.setdefault(country, {})
                    for day in daily_info[country]:
                        results[country].setdefault(day, 0)
                        results[country][day] += daily_info[
                            country][day]["count"]
        if storedfile:
            logger.info("Store the result to %s" % storedfile)
            with open(storedfile, "w") as sf:
                json.dump(results, sf)
        return results


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--folder', type=str,
                    help='input file folder: seed_xxx_document_count')
    ap.add_argument('-t', '--event', type=str, help='event type')
    ap.add_argument('-o', '--outfolder', type=str, help='outfolder')
    return ap.parse_args()


def main():
    """
    Example:
        python arabia_count.py -d /raid/tskatom/seed_protest_document_count
        -t protest -o
        /home/tskatom/workspace/icews_mode/data/features/ArabiaInform/related_doc
    """
    args = parse_args()
    # extract protest related document counts
    arabia = Arabia()
    files = glob.glob(os.path.join(args.folder, "*"))
    outfolder = os.path.join(args.outfolder,
                             datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    outfile = os.path.join(outfolder, "%s.json" % args.event)
    logger.info("Process type [%s] from folder [%s] output to [%s]"
                % (args.event, args.folder, outfile))
    arabia.e_related_docs(files, storedfile=outfile)

if __name__ == "__main__":
    sys.exit(main())
