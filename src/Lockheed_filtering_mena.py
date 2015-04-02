#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import json

MENA_COUNTRY = ["Iraq", "Egypt", "Libya", "Jordan", "Bahrain", "Syria", "Saudi Arabia"]
infile = sys.argv[1]
e_check = sys.argv[2]
basename = os.path.basename(infile)
outfolder = './LockheedMartin'
if not os.path.exists(outfolder):
    os.mkdir(outfolder)
outfile = os.path.join(outfolder, basename)
with open(infile) as rf, open(outfile, 'w') as outf:
    for l in rf:
        w = json.loads(l)
        event_type = w['eventType']
        location = w['location']
        if location[0] not in MENA_COUNTRY:
            continue
        if e_check != 'all' and event_type!="0614":
            continue
        outf.write(json.dumps(w, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    pass

