#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Consists of models for ICEWS event predicting
"""
__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
from sklearn import linear_model
import pandas as pd
import json
import numpy as np
from sklearn import preprocessing, feature_selection
from datetime import timedelta
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import statsmodels.api as sm
import argparse
np.set_printoptions(threshold=np.nan)

class Normalizer():
    def __init__(self):
        self.range_v = None
        self.min_v = None

    def fit_transform(self,data_array):
        max_v = np.max(data_array, axis=0)
        self.min_v = np.min(data_array, axis=0)
        self.range_v = max_v - self.min_v
        self.range_v[self.range_v == 0] = 1.0
        return (data_array - self.min_v) / self.range_v

    def transform(self, data_array):
        return (data_array - self.min_v) / self.range_v

    def fit(self, data_array):
        self.min_v = np.min(data_array, axis=0)
        max_v = np.max(data_array, axis=0)
        self.range_v = max_v - self.min_v

class Model(object):
    def __init__(self, name):
        self.name = name

    @staticmethod
    def oldScore(actual, predict):
        #score = 4.0 - 4.0*abs(actual-predict)/max(actual, predict, 10)
        actual = max(actual, 0)
        predict = max(predict, 0)
        occurency_score = 0.5 * ((actual>0) == (predict>0))
        accuracy_score = 3.5 - (3.5 * abs(actual - predict)) / max([actual, predict, 4])
        score = occurency_score + accuracy_score
        return score

    @staticmethod
    def oldEvaluate(truths, predictions):
        return map(Model.oldScore, truths, predictions)
    
    @staticmethod
    def score(actual, predict):
        actual = max(actual, 0)
        predict = max(predict, 0)
        occurency_score = 2.0 * ((actual>0) == (predict>0))
        accuracy_score = 2 - (2.0 * abs(actual - predict)) / max([actual, predict, 1])
        score = occurency_score + accuracy_score
        return score

    @staticmethod
    def evaluate(truths, predictions):
        paris = zip(truths, predictions)
        scores = map(Model.score, truths, predictions)
        return scores

    def sequence_train(self, trainX1, trainX2, trainY):
        """
        we first train a linear gression model on the linear model1
        compute the resudal of model1 and fit the model using feature2
        """
        #create a regression object
        regr = linear_model.LinearRegression()
        regr.fit(trainX1, trainY)
        residuals = trainY - regr.predict(trainX1)

        #train a second model based on the residuals
        lasso = linear_model.Lasso(alpha=0.1)
        lasso.fit(trainX2, residuals)

        return regr, lasso

    def sequence_predict(self, model1, model2, testX1, testX2):
        step1_prediction = model1.predict(testX1)
        step2_prediction = model2.predict(testX2)
        prediction = step1_prediction + step2_prediction
        prediction[prediction < 0.5] = 0
        prediction = map(int, prediction)
        return prediction

    def lasso_train(self, trainX, trainY):
        #directly using Lasso model on all the features
        lasso = linear_model.Lasso(alpha=0.1)
        lasso.fit(trainX, trainY)
        return lasso

    def lasso_predict(self, model, testX):
        prediction = model.predict(testX)
        prediction[prediction < 0.5] = 0
        prediction = map(int, prediction)
        return prediction

    def normal_train(self, trainX, trainY):
        #directly using Lasso model on all the features
        regr = linear_model.LinearRegression()
        regr.fit(trainX, trainY)
        return regr

    def normal_predict(self, model, testX):
        prediction = model.predict(testX)
        prediction[prediction < 0.5] = 0
        prediction = map(int, prediction)
        return prediction

    def logRg_classifier_train(self, trainX, trainY):
        logRg = linear_model.LogisticRegression(penalty="l2")
        logRg.fit(trainX, trainY)
        return logRg
    
    def logRg_classifier_predict(self, model, testX):
        predictions = model.predict(testX)
        return predictions

    def logRg_classifier(self, trainX, trainY, testX):
        if sum(trainY==0) == 0: #all the training examples are 1
            predictions = [1] * len(testX)
        else:
            model = self.logRg_classifier_train(trainX, trainY)
            predictions = self.logRg_classifier_predict(model, testX)
        return predictions

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', type=str)
    ap.add_argument('--report', type=str)
    return ap.parse_args()

def weekly_prediction_based_on_daily(daily_predictions):
    results = {}
    for i in range(len(daily_predictions)):
        date = daily_predictions.index[i]
        weekday = date.weekday()
        week_end = (date + timedelta(days=6-date.weekday())).strftime("%Y-%m-%d")
        if weekday < 6:
            results.setdefault(week_end,{})
            curr_sum = sum(daily_predictions[i-weekday:i+1])
            week_prediction = int(curr_sum * 7.0 / (weekday+1))
            results[week_end][weekday+1]=week_prediction
    return results

def weekly_prediction_use_first_two_day(event_type, icews_daily_series, daily_predictions):
    results = {}
    for i in range(len(daily_predictions)):
        date = daily_predictions.index[i]
        weekday = date.weekday()
        week_end = (date + timedelta(days=6-date.weekday())).strftime("%Y-%m-%d")
        if weekday == 1:
            #combine two series together
            t_end = date - timedelta(days=2)
            p_start = date - timedelta(days=1)
            tmp_series = icews_daily_series["2014-04-14":t_end].copy()
            tmp_series = tmp_series.append(daily_predictions[p_start:date])
            pred_start = date + timedelta(days=1)
            pred_end = date + timedelta(days=5)
            arma20 = sm.tsa.ARMA(tmp_series[:], (1,0)).fit()
            #print daily_predictions[p_start:date], p_start, date, pred_start, pred_end,t_end,tmp_series
            predictions = arma20.predict(date.strftime("%Y-%m-%d"), pred_end.strftime("%Y-%m-%d"), dynamic=True)
            predictions = predictions[1:]
            results.setdefault(week_end,{}) 
            p_v = (sum(predictions) + sum(daily_predictions[p_start:date]))
            if event_type == "14":
                p_v = p_v / 7.0
            final_prediction = abs(int(round(p_v)))
            results[week_end][weekday] = final_prediction
    return results

def dict2series(dict_obj, sample='W'):
    series = pd.Series(dict_obj)
    series.index = pd.DatetimeIndex(series.index)
    series = series.resample(sample, how='sum').fillna(0)
    return series

def dict2frame(dict_obj, sample='W'):
    days = sorted(dict_obj.keys())
    words = sorted(dict_obj[days[0]].keys())
    data_list = [dict_obj[d] for d in days]
    data_frame = pd.DataFrame(data_list, index=days, columns=words)
    data_frame.index = pd.DatetimeIndex(data_frame.index)
    data_frame = data_frame.resample(sample, how='sum').fillna(0)
    return data_frame

def getIcewsData(event_type, location, start_day, end_day, icews_count_folder):
    #get the icews count file
    icews_file = os.path.join(icews_count_folder, "%s/%s" % (event_type, location.replace(" ","_")))
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
    #get keywords count data
    keywords_file = os.path.join(keywords_count_folder, "%s_keywords_count.json" % location.replace(" ","_"))
    wordcount_obj = json.load(open(keywords_file))
    #remove those days outside the scope
    for day, word_set in wordcount_obj.items():
        if day > end_day or day < start_day:
            del wordcount_obj[day]
    return wordcount_obj


def getDocumentCount(location, start_day, end_day, document_count_folder):
    #get document count data
    document_file = os.path.join(document_count_folder, "%s_document_count.json" % location.replace(" ","_"))
    document_obj = json.load(open(document_file))
    for day, count in document_obj.items():
        if day > end_day or day < start_day:
            del document_obj[day]
    return document_obj

def transform2Binary(seriesData):
    tmpSeries = seriesData.copy()
    tmpSeries[seriesData > 0] = 1
    return tmpSeries


def weeklyCountClassifyBaseExp(event_type, location, start_day, end_day, icews_count_folder, document_count_folder, coerce_folder=None, assault_folder=None, weekly_flag=False):
    
    #get icews gsr data
    data_obj = getIcewsData(event_type, location, start_day, end_day, icews_count_folder) 
    icews_weekly_series = dict2series(data_obj, 'W')
    icews_daily_series = dict2series(data_obj, 'D')
    
    #take all three type of icews events series
    data_obj = getIcewsData("14", location, start_day, end_day, icews_count_folder) 
    icews_14_weekly_series = dict2series(data_obj, 'W')

    data_obj = getIcewsData("17", location, start_day, end_day, icews_count_folder) 
    icews_17_weekly_series = dict2series(data_obj, 'W')
    
    data_obj = getIcewsData("18", location, start_day, end_day, icews_count_folder) 
    icews_18_weekly_series = dict2series(data_obj, 'W')

    #get the documents data
    document_obj = getDocumentCount(location, start_day, end_day, document_count_folder)
    document_daily_series = dict2series(document_obj, 'D')

    #construct a weekly count classifier
    """
    Features:
        previous two weeks 14 count
        previous two weeks 17 count
        previous two weeks 18 count
        First two day's document count
    """
    train_start = "2014-04-27"
    train_end = "2014-08-17"
    test_start = "2014-08-24"
    test_end = "2014-09-14"
    
    binary_icews_weekly = transform2Binary(icews_weekly_series)
   
    #construct classifier Train phase
    cl_trainY = binary_icews_weekly[train_start:train_end]
    cl_testY = binary_icews_weekly[test_start:test_end]

    cl_trainX_14 = icews_14_weekly_series["2014-04-13":"2014-08-03"]
    cl_trainX_14 = np.c_[cl_trainX_14, icews_14_weekly_series["2014-04-20":"2014-08-10"]]
    cl_trainX_17 = icews_17_weekly_series["2014-04-13":"2014-08-03"]
    cl_trainX_17 = np.c_[cl_trainX_17, icews_17_weekly_series["2014-04-20":"2014-08-10"]]
    cl_trainX_18 = icews_18_weekly_series["2014-04-13":"2014-08-03"]
    cl_trainX_18 = np.c_[cl_trainX_18, icews_18_weekly_series["2014-04-20":"2014-08-10"]]
    
    num_train = len(cl_trainY)
    cl_trainX_doc1 = document_daily_series["2014-04-21"::7][:num_train]
    cl_trainX_doc2 = document_daily_series["2014-04-22"::7][:num_train]

    cl_trainX = np.c_[cl_trainX_14, cl_trainX_17, cl_trainX_18, cl_trainX_doc1, cl_trainX_doc2]
    #cl_trainX = np.c_[cl_trainX_14, cl_trainX_17, cl_trainX_18]

    
    #construct classifier Test Phase
    cl_testX_14 = icews_14_weekly_series["2014-08-10":"2014-08-31"]
    cl_testX_14 = np.c_[cl_testX_14, icews_14_weekly_series["2014-08-17":"2014-09-07"]]
    cl_testX_17 = icews_17_weekly_series["2014-08-10":"2014-08-31"]
    cl_testX_17 = np.c_[cl_testX_17, icews_17_weekly_series["2014-08-17":"2014-09-07"]]
    cl_testX_18 = icews_18_weekly_series["2014-08-10":"2014-08-31"]
    cl_testX_18 = np.c_[cl_testX_18, icews_18_weekly_series["2014-08-17":"2014-09-07"]]
    
    num_test = len(cl_testY)
    cl_testX_doc1 = document_daily_series["2014-08-18"::7][:num_test]
    cl_testX_doc2 = document_daily_series["2014-08-19"::7][:num_test]

    cl_testX = np.c_[cl_testX_14, cl_testX_17, cl_testX_18, cl_testX_doc1, cl_testX_doc2]
    #cl_testX = np.c_[cl_testX_14, cl_testX_17, cl_testX_18]

    #normalize the data firstly then to feature selection
    normalizer = Normalizer()
    norm_cl_trainX = normalizer.fit_transform(cl_trainX)
    norm_cl_testX = normalizer.transform(cl_testX)
    
    #do feature selection
    feature_s = SelectKBest(chi2, k=4)
    cl_trainX_new = feature_s.fit_transform(norm_cl_trainX, cl_trainY)
    cl_testX_new = feature_s.transform(norm_cl_testX)

    classifyBasedModel = Model("ClassifierBasedWeeklyModel")
    binary_predictions = classifyBasedModel.logRg_classifier(cl_trainX_new, cl_trainY.values, cl_testX_new)
    #print "%s____06%s ---> Accurcy[%0.2f]" % (location, event_type, accuracy_score(cl_testY, binary_predictions))

    #using baseline model, we predict the mean of past three weeks
    final_predictions = []
    test_days = cl_testY.index
    #for i, bin_v in enumerate(list(cl_testY.values)):
    for i, bin_v in enumerate(binary_predictions):
        day = test_days[i]
        start = day - timedelta(days=21)
        end = day - timedelta(days=7)
        if bin_v == 0:
            final_predictions.append(0)
        else:
            pred = max(1, int(np.mean(icews_weekly_series[start:end])))
            final_predictions.append(pred)

    trainY = icews_weekly_series[train_start:train_end]
    testY = icews_weekly_series[test_start:test_end]
    scores = classifyBasedModel.evaluate(testY, final_predictions)
    old_scores = classifyBasedModel.oldEvaluate(testY, final_predictions)
    return [], final_predictions, trainY, testY, scores, old_scores


def dailyLassoWithDocumentExp(event_type, location, start_day, end_day, icews_count_folder, document_count_folder, coerce_document_folder, assault_document_folder, weekly_flag=False):
    #get icews gsr data
    data_obj = getIcewsData(event_type, location, start_day, end_day, icews_count_folder) 
    icews_weekly_series = dict2series(data_obj, 'W')
    icews_daily_series = dict2series(data_obj, 'D')

    #get protest document count data
    protest_document_obj = getDocumentCount(location, start_day, end_day, document_count_folder)
    protest_document_daily_series = dict2series(protest_document_obj, 'D')
    
    #get assault document count data
    assault_document_obj = getDocumentCount(location, start_day, end_day, assault_document_folder)
    assault_document_daily_series = dict2series(assault_document_obj, 'D')
    
    #get coerce doucment count data
    coerce_document_obj = getDocumentCount(location, start_day, end_day, coerce_document_folder)
    coerce_document_daily_series = dict2series(coerce_document_obj, 'D')

    if event_type == "14":
        document_daily_series = protest_document_daily_series
    elif event_type == "17":
        document_daily_series = coerce_document_daily_series
    elif event_type == "18":
        document_daily_series = assault_document_daily_series

    #document_daily_series = protest_document_daily_series
    
    #construct protest, coerce, assault data frame
    document_daily_series = pd.DataFrame(protest_document_daily_series, columns=["protest"])
    document_daily_series["assault"] = assault_document_daily_series
    document_daily_series["coerce"] = coerce_document_daily_series
    """
    Features to use:
        1> previous two weeks's icews counts
        2> three day's documents count 
        3> three day's change percent
    """
    trainY = icews_daily_series.ix["2014-04-21":"2014-08-17"]
    testY = icews_daily_series.ix['2014-08-18':'2014-09-14']
    weeklyTestY = icews_weekly_series.ix['2014-08-24':'2014-09-14']

    #get trainX for t-1 weekly icewss count
    trainX = np.array([list(icews_weekly_series["2014-04-20":"2014-08-10"].values)]*7).T.flatten()
    #get the trainX for t-2 weekly icews count
    tmp_trainX_t2 = np.array([list(icews_weekly_series["2014-04-13":"2014-08-03"].values)]*7).T.flatten()
    #three day's documents count
    d1 = document_daily_series.ix["2014-04-21":"2014-08-17"]
    d2 = document_daily_series.ix["2014-04-20":"2014-08-16"]
    d3 = document_daily_series.ix["2014-04-19":"2014-08-15"]

    #three day's change percent
    document_daily_pct_chg = document_daily_series.pct_change()
    pct_d1 = document_daily_pct_chg.ix["2014-04-21":"2014-08-17"]
    pct_d2 = document_daily_pct_chg.ix["2014-04-20":"2014-08-16"]
    pct_d3 = document_daily_pct_chg.ix["2014-04-19":"2014-08-15"]

    #merge the trainX feature
    trainX = np.c_[trainX, tmp_trainX_t2, d1, d2, d3, pct_d1, pct_d2, pct_d3]

    #setup the test phase
    #get testX for t-1 weekly icewss count
    testX = np.array([list(icews_weekly_series["2014-08-17":"2014-09-07"].values)]*7).T.flatten()
    #get the trainX for t-2 weekly icews count
    tmp_testX_t2 = np.array([list(icews_weekly_series["2014-08-10":"2014-08-31"].values)]*7).T.flatten()
    #three day's documents count
    td1 = document_daily_series.ix['2014-08-18':'2014-09-14']
    td2 = document_daily_series.ix["2014-08-17":"2014-09-13"]
    td3 = document_daily_series.ix["2014-08-16":"2014-09-12"]

    #three day's change percent
    document_daily_pct_chg = document_daily_series.pct_change()
    tpct_d1 = document_daily_pct_chg.ix['2014-08-18':'2014-09-14']
    tpct_d2 = document_daily_pct_chg.ix["2014-08-17":"2014-09-13"]
    tpct_d3 = document_daily_pct_chg.ix["2014-08-16":"2014-09-12"]

    #merge the testx feature
    testX = np.c_[testX, tmp_testX_t2, td1, td2, td3, tpct_d1, tpct_d2, tpct_d3]

    #normalize the data firstly then to feature selection
    normalizer = Normalizer()
    norm_trainX = normalizer.fit_transform(trainX)
    norm_testX = normalizer.transform(testX)
    
    dailyLassoModel = Model("dailyLassoWithDocumentExp")
    lasso = dailyLassoModel.lasso_train(norm_trainX, trainY)
    daily_predictions = dailyLassoModel.lasso_predict(lasso, norm_testX)
    daily_scores = dailyLassoModel.evaluate(np.array(testY), daily_predictions)
    old_daily_score = dailyLassoModel.oldEvaluate(np.array(testY), daily_predictions)
    
    daily_train_predictions = dailyLassoModel.lasso_predict(lasso, norm_trainX)


    if weekly_flag:
        #construct daily prediction series
        daily_prediction_series = pd.Series(data=daily_predictions, index=testY.index)
        #weekly_result = weekly_prediction_based_on_daily(daily_prediction_series)
        weekly_result = weekly_prediction_use_first_two_day(event_type, icews_daily_series, daily_prediction_series)
        weeks = sorted(weekly_result.keys())
        #we only return best results
        best_ave_score = 0,
        best_ave_old_score = 0
        best_day = -1
        best_score = None
        best_old_score = None
        best_w_prediction = None
        for weekday in range(1,2):
            w_prediction = [weekly_result[w][weekday] for w in weeks]
            weekly_score = dailyLassoModel.evaluate(np.array(weeklyTestY), w_prediction)
            old_weekly_score = dailyLassoModel.oldEvaluate(np.array(weeklyTestY), w_prediction)
            if np.average(weekly_score) >= best_ave_score:
                best_ave_score=np.average(weekly_score)
                best_score = weekly_score
                best_old_score = old_weekly_score
                best_w_prediction = w_prediction
                best_day = weekday
        
        if location == "Iraq" and event_type == "19":
            print best_w_prediction
            print "TestWeekly", weeklyTestY
            with open('./%s_daily_pred.txt' % event_type, 'w') as dw:
                for x in daily_predictions:
                    dw.write("%d\n" % x)
            sys.exit()
        return [], best_w_prediction, trainY, weeklyTestY, best_score, best_old_score
    else:
        return daily_train_predictions, daily_predictions, trainY, testY, daily_scores, old_daily_score

def dailyLassoMoreFeatureModelExp(event_type, location, start_day, end_day, icews_count_folder, keywords_count_folder, coerce_keywords_folder, assault_keywors_folder, weekly_flag=False):
    #get icews gsr data
    data_obj = getIcewsData(event_type, location, start_day, end_day, icews_count_folder) 
    icews_weekly_series = dict2series(data_obj, 'W')
    icews_daily_series = dict2series(data_obj, 'D')

    #for different event type we will use different keywords folder
    if event_type == "14":
        pass #keep the same
    elif event_type == "17":
        keywords_count_folder = coerce_keywords_folder
    elif event_type == "18":
        keywords_count_folder = assault_keywors_folder
    #get keywords count data
    wordcount_obj = getKeywordsCount(location, start_day, end_day, keywords_count_folder)
    keyword_frame = dict2frame(wordcount_obj, 'D')

    """
    setup the training phase and testphase
    we will use t -1 and t-2 weekly's icews count as feature
    we will also use previous three day's keyword: d, d-1, d-2
    we need to very careful about the start and end of the day
    """
    trainY = icews_daily_series.ix["2014-04-21":"2014-08-17"]
    testY = icews_daily_series.ix['2014-08-18':'2014-09-14']
    weeklyTestY = icews_weekly_series.ix['2014-08-24':'2014-09-14']

    #get the trainX for t-1 weekly icews count
    trainX = np.array([list(icews_weekly_series["2014-04-20":"2014-08-10"].values)]*7).T.flatten()
    #get the trainX for t-2 weekly icews count
    tmp_trainX_t2 = np.array([list(icews_weekly_series["2014-04-13":"2014-08-03"].values)]*7).T.flatten()
    trainX = np.c_[trainX, tmp_trainX_t2]

    # get the 3 day's keywords count
    words_t = keyword_frame.ix["2014-04-21":"2014-08-17"]
    words_t1 = keyword_frame.ix["2014-04-20":"2014-08-16"]
    words_t2 = keyword_frame.ix["2014-04-19":"2014-08-15"]

    trainX = np.c_[trainX, words_t, words_t1, words_t2]

    #setup the test data
    testX = np.array([list(icews_weekly_series["2014-08-17":"2014-09-07"].values)]*7).T.flatten()
    tmp_testX_t2 = np.array([list(icews_weekly_series["2014-08-10":"2014-08-31"].values)]*7).T.flatten()

    # get the 3 day's keywords count
    words_t = keyword_frame.ix["2014-08-18":"2014-09-14"]
    words_t1 = keyword_frame.ix["2014-08-17":"2014-09-13"]
    words_t2 = keyword_frame.ix["2014-08-16":"2014-09-12"]
    testX = np.c_[testX, tmp_testX_t2,words_t, words_t1, words_t2]

    #normalize the data firstly then to feature selection
    normalizer = Normalizer()
    norm_trainX = normalizer.fit_transform(trainX)
    norm_testX = normalizer.transform(testX)

    #remove those features has very low variance
    selection = feature_selection.VarianceThreshold()
    new_norm_trainX = selection.fit_transform(norm_trainX)
    new_norm_testX = selection.transform(norm_testX)


    dailyLassoModel = Model("dailyLassoMoreFeatureModelExp")
    lasso = dailyLassoModel.lasso_train(new_norm_trainX, trainY)
    daily_predictions = dailyLassoModel.lasso_predict(lasso, new_norm_testX)
    daily_scores = dailyLassoModel.evaluate(np.array(testY), daily_predictions)
    old_daily_scores = dailyLassoModel.oldEvaluate(np.array(testY), daily_predictions)
    
    daily_train_predictions = dailyLassoModel.lasso_predict(lasso, new_norm_trainX)

    if weekly_flag:
        #construct daily prediction series
        daily_prediction_series = pd.Series(data=daily_predictions, index=testY.index)
        #weekly_result = weekly_prediction_based_on_daily(daily_prediction_series)
        weekly_result = weekly_prediction_use_first_two_day(event_type, icews_daily_series, daily_prediction_series)
        weeks = sorted(weekly_result.keys())
        #we only return best results
        best_ave_score = 0,
        best_ave_old_score = 0
        best_day = -1
        best_score = None
        best_old_score = None
        best_w_prediction = None
        for weekday in range(1,2):
            w_prediction = [weekly_result[w][weekday] for w in weeks]
            weekly_score = dailyLassoModel.evaluate(np.array(weeklyTestY), w_prediction)
            old_weekly_score = dailyLassoModel.oldEvaluate(np.array(weeklyTestY), w_prediction)
            if np.average(weekly_score) >= best_ave_score:
                best_ave_score=np.average(weekly_score)
                best_score = weekly_score
                best_old_score = old_weekly_score
                best_w_prediction = w_prediction
                best_day = weekday
        if best_day == -1:
            print location, event_type
            sys.exit()
        return [], best_w_prediction, trainY, weeklyTestY, best_score, best_old_score
    else:
        return daily_train_predictions, daily_predictions, trainY, testY, daily_scores, old_daily_scores


def dailyLassoModelExp(event_type, location, start_day, end_day, icews_count_folder, keywords_count_folder,coerce_folder=None, assault_folder=None,weekly_flag=False):
    #get icews gsr data
    data_obj = getIcewsData(event_type, location, start_day, end_day, icews_count_folder)
    icews_weekly_series = dict2series(data_obj, 'W')
    icews_daily_series = dict2series(data_obj, 'D')

    #get keywords count data
    wordcount_obj = getKeywordsCount(location, start_day, end_day, keywords_count_folder)
    keyword_frame = dict2frame(wordcount_obj, 'D')

    """
    Setup the training phase and test phase
    The test phrase range from: 2014-08-18 to 2014-09-14
    The train phrase range from: 2014-04-14 to 2014-08-17
    """
    trainY = icews_daily_series.ix["2014-04-14":"2014-08-17"].values
    testY = icews_daily_series.ix["2014-08-18":"2014-09-14"].values

    #get the trainX for t-1 weekly icews count
    trainX = np.array([list(icews_weekly_series.ix["2014-04-13":"2014-08-10"].values)]*7).transpose().flatten()
    #add the historical keyword count into feature list
    train_f_words = keyword_frame.ix["2014-04-14":"2014-08-17"]
    trainX = np.c_[trainX, train_f_words]


    testX = np.array([list(icews_weekly_series.ix["2014-08-17":"2014-09-07"].values)]*7).transpose().flatten()
    #add the historical keyword count into feature list
    test_f_words = keyword_frame.ix["2014-08-18":"2014-09-14"]
    testX = np.c_[testX, test_f_words]

    #normalize the data firstly then to feature selection
    normalizer = Normalizer()
    norm_trainX = normalizer.fit_transform(trainX)
    norm_testX = normalizer.transform(testX)

    #remove those features has very low variance
    print norm_trainX
    
    selection = feature_selection.VarianceThreshold()
    new_norm_trainX = selection.fit_transform(norm_trainX)
    new_norm_testX = selection.transform(norm_testX)
    
    dailyLassoModel = Model("dailyLassoModel")
    lasso = dailyLassoModel.lasso_train(new_norm_trainX, trainY)
    daily_predictions = dailyLassoModel.lasso_predict(lasso, new_norm_testX)
    daily_scores = dailyLassoModel.evaluate(np.array(testY), daily_predictions)
    old_daily_scores = dailyLassoModel.oldEvaluate(np.array(testY), daily_predictions)
    
    daily_train_predictions = dailyLassoModel.lasso_predict(lasso, new_norm_trainX)

    return daily_train_predictions, daily_predictions, trainY, testY, daily_scores, old_daily_scores

    #to make weekly prediction based on daily


def seqModelExp(event_type, location, start_day, end_day, icews_count_folder, keywords_count_folder, coerce_folder, assault_folder, weekly_flag=False):
    #get icews_count file
    icews_file = os.path.join(icews_count_folder, "%s/%s" % (event_type, location.replace(" ","_")))
    data_obj = {}
    with open(icews_file) as count_f:
        for line in count_f:
            d, count = line.strip().split("\t")
            if d >= start_day and d <= end_day:
                data_obj[d] = int(count)
        #make sure there are some value in the begin and end day, even for zero
        data_obj.setdefault(start_day, 0)
        data_obj.setdefault(end_day, 0)
    #construct icews_series of weekly count
    icews_series = dict2series(data_obj)

    #start to load the keywords count
    keywords_file = os.path.join(keywords_count_folder, "%s_keywords_count.json" % location.replace(" ","_"))
    #remove those days outside the scope
    wordcount_obj = json.load(open(keywords_file))
    for day, word_set in wordcount_obj.items():
        if day > end_day or day < start_day:
            del wordcount_obj[day]
    keyword_frame = dict2frame(wordcount_obj)

    #let setup the train and test phrase, we use last 4 weeks as test
    index = -4
    backward = 2
    trainY = icews_series[backward:index] #since we need to use last 2 weeks icews as feature
    testY = icews_series[index:]

    trainX1 = np.array(icews_series[:index-backward])
    for i in range(1,backward):
        trainX1 = np.c_[trainX1, np.array(icews_series[i:index-backward+i])]

    trainX2 = np.array(keyword_frame[backward-1:index-1])

    testX1 = np.array(icews_series[index-backward:-1*backward])
    for i in range(1,backward):
        testX1 = np.c_[testX1, np.array(icews_series[index-backward+i:-1*backward+i])]
    testX2 = np.array(keyword_frame[index-1:-1])

    seqModel = Model("SequenceModel")
    #normalize the data
    normalizer = Normalizer()
    norm_trainX1 = normalizer.fit_transform(trainX1)
    norm_testX1 = normalizer.transform(testX1)

    norm_trainX2 = normalizer.fit_transform(trainX2)
    norm_testX2 = normalizer.transform(testX2)

    regr, lasso = seqModel.sequence_train(norm_trainX1, norm_trainX2, trainY)
    predictions = seqModel.sequence_predict(regr, lasso, norm_testX1, norm_testX2)
    
    scores = seqModel.evaluate(np.array(testY), predictions)
    old_scores = seqModel.oldEvaluate(np.array(testY), predictions)

    #we also want to get the trainging prediction as well
    train_predictions = seqModel.sequence_predict(regr, lasso, trainX1, trainX2)
    return train_predictions, predictions, trainY, testY, scores, old_scores

def normalRegressExp(event_type, location, start_day, end_day, icews_count_folder, keywords_count_folder, coerce_folder, assault_folder, weekly_flag=False):
    #get icews_count file
    icews_file = os.path.join(icews_count_folder, "%s/%s" % (event_type, location.replace(" ", "_")))
    data_obj = {}
    with open(icews_file) as count_f:
        for line in count_f:
            d, count = line.strip().split("\t")
            if d >= start_day and d <= end_day:
                data_obj[d] = int(count)
        #make sure there are some value in the begin and end day, even for zero
        data_obj.setdefault(start_day,0)
        data_obj.setdefault(end_day,0)
    icews_series = dict2series(data_obj)

    #load the keyword count 
    keywords_file = os.path.join(keywords_count_folder, "%s_keywords_count.json" % location.replace(" ","_"))
    #remove those days outside the scope
    wordcount_obj = json.load(open(keywords_file))
    for day, word_set in wordcount_obj.items():
        if day > end_day or day < start_day:
            del wordcount_obj[day]
    keyword_frame = dict2frame(wordcount_obj)

    #construct the traning and test dataset
    index = -4
    backward = 2
    trainY = icews_series[backward:index]
    testY = icews_series[index:]
    trainX = np.array(icews_series[:index-backward])
    for i in range(1, backward):
        trainX = np.c_[trainX, np.array(icews_series[i:index-backward+i])]
    #add the keywords columns
    trainX = np.c_[trainX, np.array(keyword_frame[backward-1:index-1])]

    testX = np.array(icews_series[index-backward:-1*backward])
    for i in range(1,backward):
        testX = np.c_[testX, np.array(icews_series[index-backward+i:-1*backward+i])]
    testX = np.c_[testX, np.array(keyword_frame[index-1:-1])]

    normalModel = Model("normaoModel")
    #normalize the data
    normalizer = Normalizer()
    norm_trainX = normalizer.fit_transform(trainX)
    norm_testX = normalizer.transform(testX)

    lasso = normalModel.lasso_train(norm_trainX, trainY)
    predictions = normalModel.lasso_predict(lasso, norm_testX)

    scores = normalModel.evaluate(np.array(testY), predictions)
    old_scores = normalModel.oldEvaluate(np.array(testY), predictions)
    
    #get the training predictions
    train_predictions = normalModel.lasso_predict(lasso, trainX)
    return train_predictions, predictions, trainY, testY, scores, old_scores


def regressOnHisExp(event_type, location, start_day, end_day, icews_count_folder, keywords_count_folder, coerce_folder=None, assault_folder=None, weekly_flag=False):
    #get icews_count file
    icews_file = os.path.join(icews_count_folder, "%s/%s" % (event_type, location.replace(" ", "_")))
    data_obj = {}
    with open(icews_file) as count_f:
        for line in count_f:
            d, count = line.strip().split("\t")
            if d >= start_day and d <= end_day:
                data_obj[d] = int(count)
        #make sure there are some value in the begin and end day, even for zero
        data_obj.setdefault(start_day,0)
        data_obj.setdefault(end_day,0)
    icews_series = dict2series(data_obj)

    #construct the traning and test dataset
    index = -4
    backward = 2
    trainY = icews_series[backward:index]
    testY = icews_series[index:]
    trainX = np.array(icews_series[:index-backward])
    for i in range(1, backward):
        trainX = np.c_[trainX, np.array(icews_series[i:index-backward+i])]

    testX = np.array(icews_series[index-backward:-1*backward])
    for i in range(1,backward):
        testX = np.c_[testX, np.array(icews_series[index-backward+i:-1*backward+i])]

    normalModel = Model("regreeOnHisModel")
    normal = normalModel.normal_train(trainX, trainY)
    predictions = normalModel.normal_predict(normal, testX)

    scores = normalModel.evaluate(np.array(testY), predictions)
    old_scores = normalModel.oldEvaluate(np.array(testY), predictions)
    train_predictions = normalModel.normal_predict(normal, trainX)

    return train_predictions, predictions, trainY, testY, scores, old_scores

def experimentSetup(exp_label, algorithm, report, start_day, end_day, icews_gsr_folder, keyword_count_folder, coerce_folder=None, assault_folder=None, desc=None, weekly_flag=False):
    print "%s\n" % desc

    event_types = ["14", "17", "18"]
    countries = ["Iraq", "Egypt", "Libya", "Jordan", "Bahrain", "Syria", "Saudi Arabia"]
    report.write("-----------------------------------------------------------------\n")
    report.write("%s\n" % desc)
    report.write("%s\t0614\t0617\t0618\n" % "Country".ljust(12))

    result = {}
    tmp_result = {}
    for country in countries:
        report.write("%s" % country.ljust(12))
        for event_type in event_types:
            train_predictions, predictions, trainY, testY, scores, old_scores = algorithm(event_type, country, start_day, end_day, icews_gsr_folder, keyword_count_folder, coerce_folder, assault_folder, weekly_flag)
            score = np.average(scores)
            old_score = np.average(old_scores)
            report.write("\t%0.2f (%0.2f)" % (score, old_score))
            result.setdefault(event_type,{})
            tmp_result.setdefault(event_type,{"score":[], "old_score":[]})
            tmp_result[event_type]["score"].append(score)
            tmp_result[event_type]["old_score"].append(old_score)
            result[event_type][country] = {"trainY": list(trainY),
                    "testY": list(testY), "train_predictions":list(train_predictions),
                    "predictions": list(predictions)}
        report.write("\n")
    #write the average to the end of the line
    report.write("%s" % "Average".ljust(12))
    for event_type in event_types:
        report.write("\t%0.2f (%0.2f)" % (np.average(tmp_result[event_type]["score"]), 
            np.average(tmp_result[event_type]["old_score"])))
    report.write("\n")
    #output detail of each experiments
    with open("../report/%s" % exp_label, "w") as e:
        json.dump(result, e)

def main():
    from datetime import datetime
    args = parse_args()

    start_day = "2014-04-07"
    end_day = "2014-09-14"
    icews_gsr_folder = "../data/icews_gsr/206"
    mouna_keywords = "../data/mouna_words"
    achla_keywords = "../data/achla_words"
    achla_document = "../data/achla_document"
    mouna_document = "../data/mouna_document"
    handbook_keywords = "../data/handbook_words"
    handbook_document = "../data/handbook_document"
    mouna_coerce_keywords = "../data/coerce_keywords"
    mouna_coerce_document = "../data/coerce_document"
    mouna_assault_keywords = "../data/assault_keywords"
    mouna_assault_document = "../data/assault_document"

    icews_assault_keywords = "../data/icews_assault_keywords"
    icews_assault_document = "../data/icews_assault_document"
    
    icews_coerce_keywords = "../data/icews_coerce_keywords"
    icews_coerce_document = "../data/icews_coerce_document"
    
    report_prefix = args.report
    report = open("../report/%s_performance_%s" % (report_prefix, 
        datetime.now().strftime("%Y-%m-%d")), 'w')
    exp = args.exp

    if exp == "all" or exp == "exp1":
        """
        Experiment 1: train two model separately using handbook keywords
        """
        desc = "Experiment 1: train two models separately using handbook keywords"
        experimentSetup("Exp1", seqModelExp, report, start_day, end_day, icews_gsr_folder, handbook_keywords, desc)

    if exp == "all" or exp == "exp2":
        """
        Experiment 2: train two model separately using mouna keywords
        """
        desc = "Experiment 2: train two model separately using mouna keywords"
        experimentSetup("Exp2", seqModelExp, report, start_day, end_day, icews_gsr_folder,mouna_keywords, desc)
    
    if exp == "all" or exp == "exp3":
        """
        Experiment 3: train one model using handbook keywords
        """
        desc = "Experiment 3: train one Lasso model using handbook keywords"
        experimentSetup("Exp3", normalRegressExp, report, start_day, end_day, icews_gsr_folder,handbook_keywords, desc)
    
    if exp == "all" or exp == "exp4":
        """
        Experiment 4: train one model using mouna keywords
        """
        desc = "Experiment 4: train one Lasso model using mouna keywords"
        experimentSetup("Exp4", normalRegressExp, report, start_day, end_day, icews_gsr_folder,mouna_keywords, desc)

    if exp == "all" or exp == "exp5":
        """
        Experiment 5: train one model using only past weeks' weekly count
        """
        desc = "Experiment 5: train one model using only past weeks' weekly count"
        experimentSetup("Exp5", regressOnHisExp, report, start_day, end_day, icews_gsr_folder,mouna_keywords, desc)

    if exp == "all" or exp == "exp6":
        """
        Experiment 6: train one lasso model to make prediction in daily level using handbook keywords
            Features: 
            previous week's icews count
            previous day's keyword count
        """
        desc = "Experiment 6: train one lasso model to make prediction in daily level using Handbook keywords"
        experimentSetup("Exp6", dailyLassoModelExp, report, start_day, end_day, icews_gsr_folder,handbook_keywords, desc)
    
    if exp == "all" or exp == "exp7":
        """
        Experiment 7: train one lasso model to make prediction in daily level using mouna keywords
        Features: 
            previous week's icews count
            previous day's keyword count
        """
        desc = "Experiment 7: train one lasso model to make prediction in daily level using Mouna keywords"
        experimentSetup("Exp7", dailyLassoModelExp, report, start_day, end_day, icews_gsr_folder,mouna_keywords, desc)

    if exp == "all" or exp == "exp8":
        """
        Experiment 8: train one lasso model to make prediction in daily level using more recent factors based on achla keywords 
            Features: 
            previous two week's icews count
            previous three day's keyword count
        """
        desc = "Experiment 8: train one lasso model to make prediction in daily level using more recent factors based on handbook keywords"
        experimentSetup("Exp8", dailyLassoMoreFeatureModelExp, report, start_day, end_day, icews_gsr_folder,handbook_keywords, mouna_coerce_keywords, mouna_assault_keywords, desc)

    if exp == "all" or exp == "exp9":
        """
        Experiment 9: train one lasso model to make prediction in daily level using more recent facotrs based on mouna keywords
            Features: 
            previous two week's icews count
            previous three day's keyword count
        """
        desc = "Experiment 9: train one lasso model to make prediction in daily level using more recent factors based on Mouna keywords"
        experimentSetup("Exp9", dailyLassoMoreFeatureModelExp, report, start_day, end_day, icews_gsr_folder, mouna_keywords, mouna_coerce_keywords, mouna_assault_keywords, desc)

    if exp == "all" or exp == "exp10":
        """
        Experiment 10: train one lasso model to make prediction in daily level using documents as feature based on handbook keywords 
            Features: 
            previous two week's icews count
            previous three day's document count
            previous three day's document count change percent
        """
        desc = "Experiment 10: train one lasso model to make prediction in daily level using documents as feature based on Achla keywords"
        experimentSetup("Exp10", dailyLassoWithDocumentExp, report, start_day, end_day, icews_gsr_folder,handbook_document, desc)

    if exp == "all" or exp == "exp11":
        """
        Experiment 11: train one lasso model to make prediction in daily level using documents as feature based on mouna keywords
            Features: 
            previous two week's icews count
            previous three day's document count
            previous three day's document count change percent
        """
        desc = "Experiment 11: train one lasso model to make prediction in daily level using document as feature based on Mouna keywords"
        experimentSetup("Exp11", dailyLassoWithDocumentExp, report, start_day, end_day, icews_gsr_folder,mouna_document, desc)

    if exp == "all" or exp == "exp12":
        """
        Experiment 12: train one lasso model to make weekly prediction based on daily prediction using more recent factors based on achla keywords 
        Features: 
            previous two week's icews count
            previous three day's keyword count
        """
        desc = "Experiment 12: train one lasso model to make weekly prediction based on daily level prediction using more recent factors based on Handbook keywords"
        experimentSetup("Exp12", dailyLassoMoreFeatureModelExp, report, start_day, end_day, icews_gsr_folder,handbook_keywords, mouna_coerce_keywords, mouna_assault_keywords, desc, weekly_flag=True)

    if exp == "all" or exp == "exp13":
        """
        Experiment 13: train one lasso model to make weekly prediction based on  daily level prediction using more recent facotrs based on mouna keywords
        Features: 
            previous two week's icews count
            previous three day's keyword count
        """
        desc = "Experiment 13: train one lasso model to make weekly prediction based on daily level using more recent factors based on Mouna keywords"
        experimentSetup("Exp13", dailyLassoMoreFeatureModelExp, report, start_day, end_day, icews_gsr_folder,mouna_keywords, mouna_coerce_keywords, mouna_assault_keywords, desc, weekly_flag=True)

    if exp == "all" or exp == "exp14":
        """
        Experiment 14: train one lasso model to make weeekly prediction based on daily level using documents as feature based on achla keywords 
        Features: 
            previous two week's icews count
            previous three day's document count
            previous three day's document count change percent
        """
        desc = "Experiment 14: train one lasso model to make weekly prediction based on daily level prediction using documents as feature based on Handbook keywords"
        experimentSetup("Exp14", dailyLassoWithDocumentExp, report, start_day, end_day, icews_gsr_folder,handbook_document, mouna_coerce_document, mouna_assault_document, desc, weekly_flag=True)

    if exp == "all" or exp == "exp15":
        """
        Experiment 15: train one lasso model to make weekly prediction based on daily level prediction using documents as feature based on mouna keywords
        Features: 
            previous two week's icews count
            previous three day's document count
            previous three day's document count change percent
        """
        desc = "Experiment 15: train one lasso model to make weekly prediction based on daily level prediction using document as feature based on Mouna keywords"
        experimentSetup("Exp15", dailyLassoWithDocumentExp, report, start_day, end_day, icews_gsr_folder,mouna_document, mouna_coerce_document, mouna_assault_document, desc, weekly_flag=True)
    
    if exp == "all" or exp == "exp16":
        """
        Experiment 16: classify binary firstly and then use past three weeks mean as prediction
        Features:
            past two week's 14
            past two week's 17
            past two week's 18
            first two days' document
        """
        desc = "Experiment 16: classify binary firstly and then use past three weeks mean as prediction"
        experimentSetup("Exp16", weeklyCountClassifyBaseExp, report, start_day, end_day, icews_gsr_folder,handbook_document, desc, weekly_flag=True)
    
    if exp == "all" or exp == "exp17":
        """
        Experiment 17: train one lasso model to make weekly prediction based on daily level prediction using documents as feature based on icews keywords
        Features: 
            previous two week's icews count
            previous three day's document count
            previous three day's document count change percent
        """
        desc = "Experiment 17: train one lasso model to make weekly prediction based on daily level prediction using document as feature based on icews keywords"
        experimentSetup("Exp17", dailyLassoWithDocumentExp, report, start_day, end_day, icews_gsr_folder, handbook_document, icews_coerce_document, icews_assault_document, desc, weekly_flag=True)

    if exp == "all" or exp == "exp18":
        """
        Experiment 18: train one lasso model to make weekly prediction based on daily level prediction using keywordss as feature based on icews keywords
        Features: 
            previous two week's icews count
            previous three day's keywords count
        """
        desc = "Experiment 18: train one lasso model to make weekly prediction based on daily level prediction using keywords as feature based on icews keywords"
        experimentSetup("Exp18", dailyLassoMoreFeatureModelExp, report, start_day, end_day, icews_gsr_folder, handbook_keywords, icews_coerce_keywords, icews_assault_keywords, desc, weekly_flag=True)
    report.flush()
    report.close()

if __name__ == "__main__":
    main()
