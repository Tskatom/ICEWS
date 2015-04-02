#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import pandas as pds

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

def construct_dataset(countries, icews_folder, region="mena"):
    series = []
    for country in countries:
        coun_file = os.path.join(icews_folder, country.replace(" ", "_"))
        city = CAPITAL_COUNTRY[country]
        city_file = os.path.join(icews_folder, city.replace(" ", "_"))
        country_series = pds.Series.from_csv(coun_file, sep='\t')
        country_series.name = country
        date_range = pds.date_range('2012-01-01', '2015-03-22')
        country_series = country_series.reindex(date_range).fillna(0)

        city_series = pds.Series.from_csv(city_file, sep='\t')
        city_series.name = city
        city_series = city_series.reindex(date_range).fillna(0)

        series.append(country_series)
        series.append(city_series)

    df = pds.concat(series, axis=1)

    weekly_df = df.resample('W', how='sum').fillna(0)
    weekly_df.index.name = 'date'
    #out put to file
    outf = os.path.join('./data', '%s.csv' % region)
    weekly_df.to_csv(outf)

test = True

if __name__ == "__main__":
    if test:
        countries = MENA_COUNTRY
        icews_folder = "/raid/home/tskatom/workspace/icews_model/data/icews_gsr/232/14"
        construct_dataset(countries, icews_folder)

