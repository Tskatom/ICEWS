#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import re

in_folder = sys.argv[1]
outname = sys.argv[2]

files = os.listdir(in_folder)
anomal = 0
with open(outname, "w") as outfile:
    for f in files:
        f_an = 0
        for l in open(os.path.join(in_folder, f)):
            l_arr = l.split("\t")
            if len(l_arr) == 22:
                outfile.write(l)
            elif len(l_arr) == 24:
                outfile.write('\t'.join(l_arr[:-2]) + "\n")
            else:
                print len(l_arr)
                print l_attr
                print f
                sys.exit()
                anomal += 1
                f_an += 1
        print "Done %s , anormal %d " % (f, f_an)

print "Total Anomal %d" % anomal
if __name__ == "__main__":
    pass

