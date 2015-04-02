#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import argparse
import json
from dateutil import parser
from datetime import timedelta

CAP = {
    "Argentina": "Buenos Aires",
    "Brazil": "Distrito Federal",
    "Chile": "Santiago",
    "Colombia": "Bogota",
    "Ecuador": "Quito",
    "El Salvador": "San Salvador",
    "Mexico": "Mexico City",
    "Paraguay": "Asuncion",
    "Uruguay": "Montevideo",
    "Venezuela": "Caracas",
    "Iraq": "Baghdad",
    "Egypt": "Cairo",
    "Libya": "Tripoli",
    "Jordan": "Amman",
    "Bahrain": "Manama",
    "Syria": "Damascus",
    "Saudi Arabia": "Riyadh"
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--surr', type=str, help='the original surr file')
    p.add_argument('--version', type=int, help='the raw icews message')
    return p.parse_args()

def transform(surr_ori, icews_folder):
    basename = os.path.basename(surr_ori)
    outfile = os.path.join('./', "T_%s" % basename)
    with open(surr_ori) as surr, open(outfile, 'w') as outf:
        for line in surr:
            s_obj = json.loads(line)
            location = s_obj['location']
            summary_count = {}
            day = s_obj['day']
            for i, col in enumerate(s_obj['columns']):
                val = s_obj['values'][i]
                summary_count[col] = {day: val}
            s_obj['summaryCount'] = summary_count
            # construct the start and end day
            start = (parser.parse(day) - timedelta(days=7)).strftime('%Y-%m-%d')
            derivedFrom = {'start': start, 'end':day,
                           'source': 'ICEWS event'}
            derivedIds = []
            # extract the icews embersId for surrogate
            country = location[0]
            city = location[2]
            if city == '-':
                # this is city
                loc = CAP[country].replace(" ", "_")
                data_f = os.path.join(icews_folder, loc)
            else:
                data_f = os.path.join(icews_folder, country.replace(" ", "_"))

            with open(data_f) as df:
                for l in df:
                    l_obj = json.loads(l)
                    for e in l_obj['events']:
                        derivedIds.append(e['embersId'])
            derivedFrom["derivedIds"] = derivedIds
            s_obj["derivedFrom"] = derivedFrom

            dump_s = json.dumps(s_obj, ensure_ascii=False)
            if type(dump_s) == unicode:
                dump_s = dump_s.encode('utf-8')
            outf.write(dump_s + '\n')


def main():
    args = parse_args()
    icews_folder = os.path.join("/home/tskatom/workspace/icews_model/data/icews_gsr_region/", str(args.version))
    transform(args.surr, icews_folder)


if __name__ == "__main__":
    main()
