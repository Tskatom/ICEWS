#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import multi_task as mt
param = {"dayFile":
         "/raid/tskatom/raw_arabia_inform/arabia-2014-09-20-23-59-59",
         "keywordsFile":
         "/home/tskatom/workspace/icews_model/data/keywords/seed_assault_keywords.txt",
         "outFolder": "/home/tskatom/tmp",
         "filterFolder": "/home/tskatom"}

mt.top_document_count(param)
