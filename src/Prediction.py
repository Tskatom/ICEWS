#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
from dateutil import parser
from datetime import datetime, timedelta
import pandas as pds
import json
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm
from etool import message
import boto
import argparse
from sklearn import linear_model

np.set_printoptions(threshold=np.nan)

COUNTRY_LIST = ["Argentina", "Brazil", "Chile", "Colombia", "Ecuador",
                "El Salvador", "Mexico", "Paraguay", "Uruguay", "Venezuela",
                "Iraq", "Egypt", "Libya", "Jordan", "Bahrain",
                "Syria", "Saudi Arabia"]

MENA_COUNTRY = ["Iraq", "Egypt", "Libya", "Jordan", "Bahrain",
                "Syria", "Saudi Arabia"]

LA_COUNTRY = ["Argentina", "Brazil", "Chile", "Colombia", "Ecuador",
                "El Salvador", "Mexico", "Paraguay", "Uruguay", "Venezuela"]

CITY_LOCATION = {"Argentina": ["Argentina", "-", "Buenos Aires"],
        "Brazil": ["Brazil", "Distrito Federal", u"Brasília"],
        "Chile": ["Chile", "Santiago", "Santiago"],
        "Colombia": ["Colombia", u"Bogotá", u"Bogotá"],
        "Ecuador": ["Ecuador", "Pichincha", "Quito"],
        "El Salvador": ["El Salvador", "San Salvador", "San Salvador"],
        "Mexico": ["Mexico", u"Ciudad de México", u"Ciudad de México"],
        "Paraguay": ["Paraguay", u"Asunción", u"Asunción"],
        "Uruguay": ["Uruguay", "Montevideo", "Montevideo"],
        "Venezuela":  ["Venezuela", "Caracas", "Caracas"],
        "Iraq": ["Iraq","-","Baghdad"],
        "Egypt": ["Egypt","-","Cairo"],
        "Libya": ["Libya","-","Tripoli"],
        "Jordan": ["Jordan","-","Amman"],
        "Bahrain": ["Bahrain","-","Manama"],
        "Syria": ["Syria","-","Damascus"],
        "Saudi Arabia": ["Saudi Arabia","-","Riyadh"]
        }

COUNTRY_LOCATION = {"Argentina": ["Argentina", "-", "-"],
        "Brazil": ["Brazil", "-", "-"],
        "Chile": ["Chile", "-", "-"],
        "Colombia": ["Colombia", "-", "-"],
        "Ecuador": ["Ecuador", "-", "-"],
        "El Salvador": ["El Salvador", "-", "-"],
        "Mexico": ["Mexico", "-", "-"],
        "Paraguay": ["Paraguay", "-", "-"],
        "Uruguay": ["Uruguay", "-", "-"],
        "Venezuela":  ["Venezuela", "-", "-"],
        "Iraq": ["Iraq","-","-"],
        "Egypt": ["Egypt","-","-"],
        "Libya": ["Libya","-","-"],
        "Jordan": ["Jordan","-","-"],
        "Bahrain": ["Bahrain","-","-"],
        "Syria": ["Syria","-","-"],
        "Saudi Arabia": ["Saudi Arabia","-","-"]
        }

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

class Normalizer():
    def __init__(self):
        self.range_v = None
        self.min_v = None

    def fit_transform(self, data_array):
        max_v = np.max(data_array, axis=0)
        self.min_v = np.min(data_array, axis=0)
        self.range_v = max_v - self.min_v
        self.range_v[self.range_v == 0] = 1.0
        return (data_array - self.min_v) /  self.range_v

    def transform(self, data_array):
        return (data_array - self.min_v) /  self.range_v

    def fit(self, data_array):
        self.min_v = np.min(data_array, axis=0)
        max_v = np.max(data_array, axis=0)
        self.range_v = max_v - self.min_v

class Mena():
    def __init__(self, location, country, pred_level):
        self.location = location
        self.country = country
        self.pred_level = pred_level

    def __score(self, actual, predict):
        actual = max(actual, 0)
        prediction = max(prediction, 0)

        occurency_score = 2.0 * ((actual>0) == (predict>0))
        accuracy_score = 2 - (2.0 * abs(actual - predict)) / max([actual, predict, 5])

        return occurency_score + accuracy_score

    def evaluate(self, truths, predictions):
        scores = map(self.__score, truths, predictions)
        return scores

    def twoStepAR(self, event_type, event_date, icews_gsr, protest_folder,
            coerce_folder, assault_folder):
        """
        Two Step AR Algorithm:
            1> make daily prediction based on document count
            2> Using AR model to predict the remaining days of the week
        """
        location = self.location
        country = self.country
        pred_level = self.pred_level
        start_day = "2014-09-01"
        train_end_day = (parser.parse(event_date) -
                         timedelta(days=3)).strftime("%Y-%m-%d")
        end_day = (parser.parse(event_date) -
                   timedelta(days=2)).strftime("%Y-%m-%d")

        #load icews series
        icews_data_obj = getIcewsData(event_type, location, start_day,
                                      end_day, icews_gsr)
        icews_weekly_series = dict2series(icews_data_obj, 'W')
        icews_daily_series = dict2series(icews_data_obj, 'D')

        #get protest document counts
        protest_document_obj = getDocumentCount(country, start_day,
                                                end_day, protest_folder)
        protest_document_daily_series = dict2series(protest_document_obj, 'D')

        #get coerce document counts
        coerce_document_obj = getDocumentCount(country, start_day,
                                               end_day, coerce_folder)
        coerce_document_daily_series = dict2series(coerce_document_obj, 'D')

        #get assault document counts
        assault_document_obj = getDocumentCount(country, start_day,
                                                end_day, assault_folder)
        assault_document_daily_series = dict2series(assault_document_obj, 'D')

        #combine three type of documents into one dataframe
        document_daily_series = pds.DataFrame(protest_document_daily_series,
                                             columns=["protest"])
        document_daily_series["assault"] = assault_document_daily_series
        document_daily_series["coerce"] = coerce_document_daily_series

        time_window = 3
        trainX = None
        for i in range(time_window):
            train_start = (parser.parse(start_day) +
                           timedelta(days=i)).strftime("%Y-%m-%d")
            train_end = (parser.parse(train_end_day) -
                         timedelta(days=time_window-i-1)).strftime("%Y-%m-%d")
            if trainX is None:
                trainX = np.c_[document_daily_series.ix[train_start:train_end]]
            else:
                trainX = np.c_[trainX, document_daily_series.ix[train_start:train_end]]

        #setup the training Y start day
        train_start = (parser.parse(start_day) +
                       timedelta(days=time_window-1)).strftime("%Y-%m-%d")
        trainY = icews_daily_series.ix[train_start:train_end_day]

        #construct test data set
        test_start = (parser.parse(end_day) -
                      timedelta(days=time_window-1)).strftime("%Y-%m-%d")
        testX = np.c_[document_daily_series.ix[test_start:end_day]].flatten()

        #normalize the data
        normalizer = Normalizer()
        norm_trainX = normalizer.fit_transform(trainX)
        norm_testX = normalizer.transform(testX)

        #fit the daily
        lasso = linear_model.Lasso(alpha=0.3)
        lasso.fit(norm_trainX, trainY)
        prediction = lasso.predict(norm_testX)
        if prediction < 0.5:
            prediction = 0
        prediction = int(prediction)


        #construct a AR model to predict the following days of a week
        tmp_series = icews_daily_series.ix[start_day:train_end_day].copy()
        tmp_series[end_day] = prediction
        tmp_series.index = pds.DatetimeIndex(tmp_series.index)
        try:
            arma50 = sm.tsa.ARMA(tmp_series[:], (1,0)).fit()
        except:
            print location, country,'======'
            sys.exit()
        week_end = (parser.parse(event_date) +
                    timedelta(days=4)).strftime("%Y-%m-%d")

        arma_predictions = arma50.predict(end_day, week_end, dynamic=True).values.tolist()
        arma_predictions = map(int, arma_predictions)

        weekly_prediction = prediction + sum(arma_predictions[-6:])

        #start to wrap the surrogate
        surrogate = wrap_twostepar_surr(icews_daily_series, protest_document_daily_series,
                coerce_document_daily_series, assault_document_daily_series)
        #start to wrap the warning
        if pred_level == "country":
            event_location = COUNTRY_LOCATION[country]
        elif pred_level == "city":
            event_location = CITY_LOCATION[country]
        else:
            print "Please Enter Correct prediction Level [%s]" % pred_level
            sys.exit()
        model_desc = "Two Step Auto Regressive Model"
        comment = "ICEWS %s Prediction use %s " % (event_type, model_desc)

        warning = wrap_warning(surrogate, event_type, weekly_prediction,
                event_location, event_date, model_desc, comment)

        return warning, surrogate

class LA():
    def __init__(self, location, country, pred_level):
        self.location = location
        self.country = country
        self.pred_level = pred_level

    def lasso(self, event_type, event_date, icews_gsr,icews_count_folder, pp_count_folder):
        location = self.location
        country = self.country
        pred_level = self.pred_level

        #load the icews weekly count
        start_day = "2012-01-01"
         #The last weekend of current event date
        end_day = (parser.parse(event_date) - timedelta(days=3)).strftime("%Y-%m-%d")

        icews_data_obj = getIcewsData(event_type, location, start_day,
                end_day, icews_gsr)
        icews_weekly_series = dict2series(icews_data_obj, 'W')

        #get icews version
        if icews_gsr[-1] == "/":
            gsr_version = os.path.basename(icews_gsr[:-1])
        else:
            gsr_version = os.path.basename(icews_gsr)

        #load the icews features: use previous weeks data as feature
        if pred_level == "city":
            icews_file = "%s-%s_icews_parent_event_counts_2012_%s.csv" % (country.lower().replace(" ", "_"),
                    location.lower().replace(" ", "_"),
                    gsr_version)
        elif pred_level == "country":
            icews_file = "%s_icews_parent_event_counts_2012_%s.csv" % (country.lower().replace(" ", "_"),
                    gsr_version)
        icews_file = os.path.join(icews_count_folder, icews_file)
        icews_data_frame = pds.DataFrame.from_csv(icews_file, sep='\t', index_col=1)
        icews_data_frame.index = pds.DatetimeIndex(icews_data_frame.index)
        icews_data_frame = icews_data_frame[start_day:].ix[:,1:].resample("W", how="sum").fillna(0)
        #print icews_data_frame

        #load the pp count:using the same week to train, since only country level
        pp_file = "%s_weeklyseries.csv" % (country.lower())
        pp_file = os.path.join(pp_count_folder, pp_file)
        pp_weekly_series = pds.Series.from_csv(pp_file, sep=',', index_col=0)
        pp_weekly_series.index = pds.DatetimeIndex(pp_weekly_series.index)

        #construnct train set
        trainY = icews_weekly_series["2013-01-13":end_day]
        #end of the trainX icews day is one week before the end_day
        trainX_ic_endday = (parser.parse(event_date) - timedelta(days=10)).strftime("%Y-%m-%d")
        trainX_ic = icews_data_frame["2013-01-06":trainX_ic_endday].values
        trainX_pp = pp_weekly_series["2013-01-13":end_day].values
        #print trainX_ic.shape, trainX_pp.shape, location
        #print icews_data_frame["2013-01-06":trainX_ic_endday]
        trainX = np.c_[trainX_ic, trainX_pp]
        print  "------------>", location
        testX_ic = icews_data_frame.ix[end_day].values
        #end of the testX planned protest day is the current weekend
        test_pp_day = (parser.parse(event_date) + timedelta(days=4)).strftime("%Y-%m-%d")
        testX_pp = pp_weekly_series[test_pp_day]
        testX = np.append(testX_ic, testX_pp)
        #print testX
        #print trainX.shape, trainY.shape
        #print trainY
        #sys.exit()
        lasso = linear_model.Lasso(alpha=0.1)
        lasso.fit(trainX, trainY)

        prediction = lasso.predict(testX)
        prediction = int(prediction) if prediction > 0 else 0

        surrogate = wrap_lasso_surr(icews_data_frame, pp_weekly_series)

        model_desc = "LA: LASOO Model"
        comment = "LASSO MODEL for LA ICEWS weekly Prediction"

        if pred_level == "country":
            event_location = COUNTRY_LOCATION[country]
        elif pred_level == "city":
            event_location = CITY_LOCATION[country]
        else:
            print "Please Enter Correct prediction Level [%s]" % pred_level
            sys.exit()

        warning = wrap_warning(surrogate, event_type, prediction, event_location,
                event_date, model_desc, comment)
        return warning, surrogate

def wrap_warning(surrogate, event_type, prediction, location, event_date,
                 model_desc, comment):
    count_warning = {}
    count_warning["derivedFrom"] = {
            "derivedIds": [surrogate["embersId"]]
            }
    count_warning["confidence"] = 1.0
    model = model_desc
    comments = comment
    count_warning["comments"] = comments
    count_warning["eventType"] = event_type
    count_warning["eventDate"] = event_date
    count_warning["warningUpdate"] = None
    count_warning["version"] = "1.0.0"
    count_warning["location"] = location
    count_warning["model"] = model
    count_warning["population"] = prediction
    count_warning["confidenceIsProbability"] = False
    count_warning["date"] = datetime.utcnow().isoformat()

    #check whether this warning will be a update or not
    conn = boto.connect_sdb()
    domain = conn.lookup("warnings")
    sql = "select * from warnings where eventType='%s' and eventDate='%s' and location='%s' order by eventDate desc" % (event_type, event_date, json.dumps(location, ensure_ascii=False))
    rs = domain.select(sql)
    rs = [r for r in rs]
    if len(rs) == 0:
        count_warning["warningUpdate"] = None
    else:
        #there are already a prediction
        print "Warn In DB ", int(rs[0]["population"]), "new ", prediction, location
        if prediction == int(rs[0]["population"]):
            print "The Same Prediction", location
            return None
        else:
            count_warning["warningUpdate"] = rs[0]["embersId"]

    count_warning = message.add_embers_ids(count_warning)
    return count_warning

def series2dict(series):
    temp_dict = series.to_dict()
    result_dict = {k.strftime('%Y-%m-%d'):int(v) for k,v in temp_dict.items()}
    return result_dict

def dataframe2dict(dataframe):
    temp_dict = dataframe.to_dict()
    result_dict = {}
    for key in temp_dict:
        result_dict[key] = {k.strftime("%Y-%m-%d"):int(v) for k,v in temp_dict[key].items()}
    return result_dict

def wrap_twostepar_surr(protest_doc, coerce_doc, assault_doc, icews_series):
    surrogate = {}
    surrogate["icews_daily"] = series2dict(icews_series)
    surrogate["protest_doc_daily"] = series2dict(protest_doc)
    surrogate["coerce_doc_daily"] = series2dict(coerce_doc)
    surrogate["assault_doc_daily"] = series2dict(assault_doc)
    surrogate["date"] = datetime.utcnow().isoformat()
    surrogate = message.add_embers_ids(surrogate)
    return surrogate

def wrap_lasso_surr(icews_dataframe, pp_series):
    surrogate = {}
    surrogate["icews_dataframe"] = dataframe2dict(icews_dataframe)
    surrogate["planned_protest"] = series2dict(pp_series)
    surrogate["date"] = datetime.utcnow().isoformat()
    surrogate = message.add_embers_ids(surrogate)
    return surrogate

def dict2series(data_obj, resample='W'):
    series = pds.Series(data_obj)
    series.index = pds.DatetimeIndex(series.index)
    series = series.resample(resample, how='sum').fillna(0)
    return series

def dict2frame(data_obj, resample='W'):
    days = sorted(dict_obj.keys())
    columns = sorted(data_obj[days[0]].keys())
    data_list = [data_obj[d] for d in days]
    data_frame = pds.DataFrame(data_list, index=days, columns=columns)
    data_frame.index = pds.DatetimeIndex(data_frame.index)
    data_frame = data_frame.resample(resample, how='sum').fillna(0)
    return data_frame

def getIcewsData(event_type, location, start_day, end_day, icews_count_folder):
    icews_file = os.path.join(icews_count_folder,
            "%s/%s" % (event_type[2:], location.replace(" ","_")))
    data_obj = {}
    with open(icews_file) as count_f:
        for line in count_f:
            d, count = line.strip().split("\t")
            if d >= start_day and d <= end_day:
                data_obj[d] = int(count)
        data_obj.setdefault(start_day, 0)
        data_obj.setdefault(end_day, 0)
    return data_obj

def getKeywordsCount(location, start_day, end_day, keywords_count_folder):
    keywords_file = os.path.join(keywords_count_folder,
                                 "%s_keywords_count.json" %
                                 location.replace(" ", "_"))
    wordcount_obj = json.load(open(keywords_file))
    #remove those days outside the scope
    for day, word_set in wordcount_obj.items():
        if day > end_day or day < start_day:
            del wordcount_obj[day]
    return wordcount_obj

def getDocumentCount(location, start_day, end_day, document_count_folder):
    #get document count data
    document_file = os.path.join(document_count_folder,
                                 "%s_document_count.json"
                                 % location.replace(" ", "_"))

    document_obj = json.load(open(document_file))

    #remove those days outside the scope
    for day, count in document_obj.items():
        if day > end_day or day < start_day:
            del document_obj[day]
    document_obj.setdefault(start_day, 0)
    document_obj.setdefault(end_day, 0)
    return document_obj

def predict_mena(args):
    event_date = args.event_date
    icews_gsr = args.icews_gsr
    protest_folder = args.protest
    coerce_folder = args.coerce
    assault_folder = args.assault
    out_folder = args.out

    currtime = datetime.utcnow().strftime("%Y-%m-%dT%H_%M_%S")
    warn_file = os.path.join(out_folder, "WARN_MENA_ICEWS_17_18_%s" % currtime)
    surr_file = os.path.join(out_folder, "SURR_MENA_ICEWS_17_18_%s" % currtime)

    wf = open(warn_file, "w")
    sf = open(surr_file, "w")

    event_types = ["0614", "0617", "0618"]
    for country in MENA_COUNTRY:
        for event_type in event_types:
            country_mena = Mena(country, country, "country")
            city_location =  CAPITAL_COUNTRY[country]
            city_mena = Mena(city_location, country, "city")
            try:
                country_warn, country_surr = country_mena.twoStepAR(event_type,
                        event_date, icews_gsr, protest_folder, coerce_folder,
                        assault_folder)

                city_warn, city_surr = city_mena.twoStepAR(event_type,
                        event_date, icews_gsr, protest_folder, coerce_folder,
                        assault_folder)
            except:
                print country, '====', city_location
                sys.exit()
            if country_warn:
                wf.write(json.dumps(country_warn, ensure_ascii=False) + "\n")
                sf.write(json.dumps(country_surr, ensure_ascii=False) + "\n")
            if city_warn:
                wf.write(json.dumps(city_warn, ensure_ascii=False) + "\n")
                sf.write(json.dumps(city_surr, ensure_ascii=False) + "\n")
    wf.flush()
    sf.flush()
    wf.close()
    sf.close()

def predict_la(args):
    event_date = args.event_date
    icews_gsr = args.icews_gsr
    planned_protest_folder = args.planned_protest
    icews_count_folder = args.icews_count
    out_folder = args.out

    currtime = datetime.utcnow().strftime("%Y-%m-%dT%H_%M_%S")
    warn_file = os.path.join(out_folder, "WARN_LA_ICEWS_17_18_%s" % currtime)
    surr_file = os.path.join(out_folder, "SURR_LA_ICEWS_17_18_%s" % currtime)

    wf = open(warn_file, "w")
    sf = open(surr_file, "w")

    event_types = ["0614", "0617", "0618"]
    for country in LA_COUNTRY:
        for event_type in event_types:
            country_la = LA(country, country, "country")
            city_location =  CAPITAL_COUNTRY[country]
            city_la = LA(city_location, country, "city")

            country_warn, country_surr = country_la.lasso(event_type,
                    event_date, icews_gsr, icews_count_folder, planned_protest_folder)

            city_warn, city_surr = city_la.lasso(event_type,
                    event_date, icews_gsr, icews_count_folder, planned_protest_folder)

            if country_warn:
                wf.write(json.dumps(country_warn, ensure_ascii=False).encode('utf-8') + "\n")
                sf.write(json.dumps(country_surr, ensure_ascii=False).encode('utf-8') + "\n")
            if city_warn:
                wf.write(json.dumps(city_warn, ensure_ascii=False).encode('utf-8')+ "\n")
                sf.write(json.dumps(city_surr, ensure_ascii=False) + "\n")
    wf.flush()
    sf.flush()
    wf.close()
    sf.close()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default="../data/warn_surr", help="output folder")
    ap.add_argument('--assault', type=str,
            default='../data/assault_document', help="assault doc folder")
    ap.add_argument('--coerce', type=str,
            default='../data/coerce_document', help='coearce doc folder')
    ap.add_argument('--protest', type=str,
            default='../data/handbook_document', help='protest doc folder')
    ap.add_argument("--planned_protest", type=str,
            default="../data/pp_counts/icews_weeklyseries")
    ap.add_argument("--icews_count", type=str, help="icews count") #format ../data/icews_counts/215
    ap.add_argument('--icews_gsr', type=str, help='icews gsr folder')
    ap.add_argument('--event_date', type=str, help="event_date")
    ap.add_argument('--mena', action='store_true')
    ap.add_argument('--la', action='store_true')
    args = ap.parse_args()
    return args

def main():
    args = parse_args()
    if args.icews_count is None:
        print "Please enter ICEWS Counts folder!"
        sys.exit()
    if args.icews_gsr is None:
        print "Please enter ICEWS Gsr folder!"
        sys.exit()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    event_date = args.event_date
    #if the event_date is not wed, then print and exit
    weekday = parser.parse(event_date).weekday()
    if weekday != 2:
        print "%s is not Wendnesday" % event_date
        sys.exit()

    if args.mena:
        predict_mena(args)

    if args.la:
        predict_la(args)


if __name__ == "__main__":
    sys.exit(main())
