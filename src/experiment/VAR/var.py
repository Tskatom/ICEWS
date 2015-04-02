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
    country_series = []
    city_series = []
    for country in countries:
        coun_file = os.path.join(icews_folder, country.replace(" ", "_"))
        city = CAPITAL_COUNTRY[country]
        city_file = os.path.join(icews_folder, city.replace(" ", "_"))

        coun_series = pds.Series.from_csv(coun_file, sep='\t')
        coun_series.name = country
        date_range = pds.date_range('2012-01-01', '2015-03-22')
        coun_series = coun_series.reindex(date_range).fillna(0)

        cy_series = pds.Series.from_csv(city_file, sep='\t')
        cy_series.name = city
        cy_series = cy_series.reindex(date_range).fillna(0)

        country_series.append(coun_series)
        city_series.append(cy_series)

    country_df = pds.concat(country_series, axis=1)
    city_df = pds.concat(city_series, axis=1)

    country_weekly_df = country_df.resample('W', how='sum').fillna(0)
    country_weekly_df.index.name = 'date'

    city_weekly_df = city_df.resample('W', how='sum').fillna(0)
    city_weekly_df.index.name = 'date'
    #out put to file
    country_outf = os.path.join('./data', 'country_%s.csv' % region)
    country_weekly_df.to_csv(country_outf)

    city_outf = os.path.join('./data', 'city_%s.csv' % region)
    city_weekly_df.to_csv(city_outf)

def score(pred, truth):
    occu = 0.5 * ( (pred > 0) == (truth > 0))
    accu = 3.5 * (1 - 1.0*abs(pred - truth)/(max([pred, truth, 4])))
    return occu + accu

def evaluate(pred_file, truth_file):
    preds = pds.DataFrame.from_csv(pred_file, sep=',', header=0, index_col=None)
    truths = pds.DataFrame.from_csv(truth_file, sep=',', header=0, index_col=None)
    names = preds.columns
    for name in names:
        p = preds[name]
        t = truths[name]
        scores = map(score, p.values, t.values)
        print np.mean(scores)

test = True

if __name__ == "__main__":
    if test:
        countries = MENA_COUNTRY
        icews_folder = "/raid/home/tskatom/workspace/icews_model/data/icews_gsr/232/14"
        construct_dataset(countries, icews_folder)
        #evaluate('./data/predictions.csv', './data/testY.csv')

