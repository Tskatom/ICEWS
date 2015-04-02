#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import json
files = os.listdir(sys.argv[1])
w = open('Historical_ICEWS_2015-03-30.json', 'w')
for f in files:
    ff = os.path.join(sys.argv[1], f)
    with open(ff) as fff:
        for l in fff:
            j = json.loads(l)
            for e in j['events']:
                ds = json.dumps(e, ensure_ascii=False)
                if type(ds) == unicode:
                    ds = ds.encode('utf-8')
                w.write(ds + '\n')

w.flush()
w.close()
if __name__ == "__main__":
    pass

