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


class Model(object):
    def __init__(self, name):
        self.name = name
    
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

def dict2series(dict_obj):
    series = pd.Series(dict_obj)
    series.index = pd.DatetimeIndex(series.index)
    series = series.resample('W', how='sum').fillna(0)
    return series

def dict2frame(dict_obj):
    days = sorted(dict_obj.keys())
    words = sorted(dict_obj[days[0]].keys())
    data_list = [dict_obj[d] for d in days]
    data_frame = pd.DataFrame(data_list, index=days, columns=words)
    data_frame.index = pd.DatetimeIndex(data_frame.index)
    data_frame = data_frame.resample('W', how='sum').fillna(0)
    return data_frame

def seqModelExp(event_type, location, start_day, end_day, icews_count_folder, keywords_count_folder):
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
    regr, lasso = seqModel.sequence_train(trainX1, trainX2, trainY)
    predictions = seqModel.sequence_predict(regr, lasso, testX1, testX2)
    
    scores = seqModel.evaluate(np.array(testY), predictions)

    #we also want to get the trainging prediction as well
    train_predictions = seqModel.sequence_predict(regr, lasso, trainX1, trainX2)
    return train_predictions, predictions, trainY, testY, scores

def normalRegressExp(event_type, location, start_day, end_day, icews_count_folder, keywords_count_folder):
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
    lasso = normalModel.lasso_train(trainX, trainY)
    predictions = normalModel.lasso_predict(lasso, testX)

    scores = normalModel.evaluate(np.array(testY), predictions)
    
    #get the training predictions
    train_predictions = normalModel.lasso_predict(lasso, trainX)
    return train_predictions, predictions, trainY, testY, scores


def regressOnHisExp(event_type, location, start_day, end_day, icews_count_folder, keywords_count_folder):
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
    train_predictions = normalModel.normal_predict(normal, trainX)

    return train_predictions, predictions, trainY, testY, scores


def main():
    from datetime import datetime

    event_types = ["14", "17", "18"]
    countries = ["Iraq", "Egypt", "Libya", "Jordan", "Bahrain", "Syria", "Saudi Arabia"]
    report = open("../report/performance_%s" % datetime.now().strftime("%Y-%m-%d"), 'w')
    """
    Experiment 1: train two model separately using achla keywords
    """
    print "Experiment 1"
    report.write("-----------------------------------------------------------------\n")
    report.write("Experiment 1: train two models separately using achla keywords\n")
    report.write("%s\t0614\t0617\t0618\n" % "Country".ljust(12))

    result = {}
    for country in countries:
        report.write("%s" % country.ljust(12))
        for event_type in event_types:
            train_predictions, predictions, trainY, testY, scores = seqModelExp(event_type, country, "2014-04-07", "2014-09-14", "../data/icews_gsr/206/", "../data/achla_words/")
            score = np.average(scores)
            report.write("\t%0.2f" % score)
        report.write("\n")

    """
    Experiment 2: train two model separately using mouna keywords
    """
    print "Experiment 2"
    report.write("-----------------------------------------------------------------\n")
    report.write("Experiment 2: train two model separately using mouna keywords\n")
    report.write("%s\t0614\t0617\t0618\n" % "Country".ljust(12))

    result = {}
    for country in countries:
        report.write("%s" % country.ljust(12))
        for event_type in event_types:
            train_predictions, predictions, trainY, testY, scores = seqModelExp(event_type, country, "2014-04-07", "2014-09-14", "../data/icews_gsr/206/", "../data/mouna_words/")
            score = np.average(scores)
            report.write("\t%0.2f" % score)
        report.write("\n")
    
    """
    Experiment 3: train one model using achla keywords
    """
    print "Experiment 3"
    report.write("-----------------------------------------------------------------\n")
    report.write("Experiment 3: train one Lasso model using achla keywords\n")
    report.write("%s\t0614\t0617\t0618\n" % "Country".ljust(12))

    result = {}
    for country in countries:
        report.write("%s" % country.ljust(12))
        for event_type in event_types:
            train_predictions, predictions, trainY, testY, scores = normalRegressExp(event_type, country, "2014-04-07", "2014-09-14", "../data/icews_gsr/206/", "../data/achla_words/")
            score = np.average(scores)
            report.write("\t%0.2f" % score)
            result.setdefault(event_type,{})
            result[event_type][country] = {"trainY": list(trainY), 
                    "testY": list(testY), "train_predictions":list(train_predictions),
                    "predictions": list(predictions)}
        report.write("\n")
    with open("../report/Exp3.result", "w") as e3:
        json.dump(result, e3)
    
    """
    Experiment 4: train one model using mouna keywords
    """
    print "Experiment 4"
    report.write("-----------------------------------------------------------------\n")
    report.write("Experiment 4: train one Lasso model using mouna keywords\n")
    report.write("%s\t0614\t0617\t0618\n" % "Country".ljust(12))

    result = {}
    for country in countries:
        report.write("%s" % country.ljust(12))
        for event_type in event_types:
            train_predictions, predictions, trainY, testY, scores = normalRegressExp(event_type, country, "2014-04-07", "2014-09-14", "../data/icews_gsr/206/", "../data/mouna_words/")
            score = np.average(scores)
            report.write("\t%0.2f" % score)
        report.write("\n")

    """
    Experiment 5: train one model using only past weeks' weekly count
    """
    print "Experiment 5"
    report.write("-----------------------------------------------------------------\n")
    report.write("Experiment 5: train one model using only past weeks' weekly count\n")
    report.write("%s\t0614\t0617\t0618\n" % "Country".ljust(12))

    result = {}
    for country in countries:
        report.write("%s" % country.ljust(12))
        for event_type in event_types:
            train_predictions, predictions, trainY, testY, scores = regressOnHisExp(event_type, country, "2014-04-07", "2014-09-14", "../data/icews_gsr/206/", "../data/mouna_words/")
            score = np.average(scores)
            report.write("\t%0.2f" % score)
        report.write("\n")
if __name__ == "__main__":
    main()
