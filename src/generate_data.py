#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import pandas as pds
import numpy as np
import json

"""
VAR model experiments
"""
MENA_COUNTRY = ["Iraq", "Egypt", "Libya", "Jordan", "Bahrain",
                "Syria", "Saudi Arabia"]

LA_COUNTRY = ["Argentina", "Brazil", "Chile", "Colombia", "Ecuador",
              "El Salvador", "Mexico", "Paraguay", "Uruguay", "Venezuela"]


CAPITAL_COUNTRY = {
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

def generate_count(country, event_type, infolder, outfolder):
    #for each region we will create one folder
    city = CAPITAL_COUNTRY[country]
    folder_names = {country:country.replace(" ", "_"),
                    city: "%s-%s" % (country.replace(" ","_"),
                                     city.replace(" ", "_"))}
    event_names = {"14": "Protest", "17": "Coerce", "18": "Assault"}
    regions = [country, city]
    for region in regions:
        f_name = os.path.join(outfolder, folder_names[region])
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        event_file = os.path.join(infolder, "%s/%s" % (event_type, region.replace(" ", "_")))
        series = pds.Series.from_csv(event_file, sep='\t', index_col=0, header=None)
        date_range = pds.date_range('1991-01-01', '2015-03-29')
        series = series.reindex(date_range).fillna(0)
        series = series.resample('D').fillna(0)
        series.index.name = 'date'
        series.name = 'count'
        series = series.astype(int)
        outfilename = os.path.join(f_name, "%s_event_count" % event_names[event_type])
        series.to_csv(outfilename, sep='\t', header=True, index_label='date')

def generate_keywords(event_type, infolder, outfolder):
    event_names = {"14": "Protest", "17": "Coerce", "18": "Assault"}
    files = os.listdir(infolder)
    keywords_count = {}
    for f in files:
        full_f = os.path.join(infolder, f)
        k_json = json.load(open(full_f))
        for country, words_set in k_json.items():
            if country not in MENA_COUNTRY:
                continue
            keywords_count.setdefault(country, {})
            for word, day_set in words_set.items():
                keywords_count[country].setdefault(word, {})
                for day, count in day_set.items():
                    keywords_count[country][word].setdefault(day, 0)
                    keywords_count[country][word][day] += count
    # for each country we construct a dataframe
    for country, words_count in keywords_count.items():
        df = pds.DataFrame(words_count)
        df.index = pds.DatetimeIndex(df.index)
        date_range = pds.date_range('2014-04-01', '2015-04-02')
        df = df.reindex(date_range).fillna(0)
        df = df.sort_index()
        # we need to generate tow files one for country and one for city
        city = CAPITAL_COUNTRY[country]
        folder_names = {country:country.replace(" ", "_"),
                        city: "%s-%s" % (country.replace(" ","_"),
                                         city.replace(" ", "_"))}
        for name in folder_names:
            f_name = os.path.join(outfolder, name)
            if not os.path.exists(f_name):
                os.mkdir(f_name)

            outfile = os.path.join(f_name, "%s_%s_keywords" % (country, event_names[event_type]))
            df = df.astype(int)
            df.to_csv(outfile, sep='\t', header=True, index_label="date", encoding='utf-8')


if __name__ == "__main__":
    #generate gsr count
    icews_folder = "/home/tskatom/workspace/icews_model/data/icews_gsr/233"
    outfolder = "/home/tskatom/workspace/icews_model/data/to_zheng"
    events = ["14", "17", "18"]
    for country in MENA_COUNTRY:
        for event_type in events:
            generate_count(country, event_type, icews_folder, outfolder)

    keywords_folders = {"14":"/raid/tskatom/arabia_mouna_keywords_count",
                        "17":"/raid/tskatom/coerce_keywords_count",
                        "18": "/raid/tskatom/assault_keywords_count"}
    #generate keywords count
    for event_type in events:
        generate_keywords(event_type, keywords_folders[event_type], outfolder)



