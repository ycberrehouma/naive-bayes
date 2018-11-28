# Data Mining Class Project
# Choice 2. Implement Naive Bayes Classifier from scratch to predict if a person makes over 50K a year.
# Yassine Berrehouma

import csv
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy

def dataCleaning(file):
    dataframe = pd.read_csv(file,header=None)
    dataframe.columns = ['age','workclass','fnlwgt','education','educationalnum','maritalstatus','occupation','relationship','race','gender','capitalgain','capitalloss','hoursperweek','nativecountry','income']

    # Encode Data
    dataframe.replace(('?'),(numpy.nan),inplace=True)
    dataframe.dropna(inplace=True)
    dataframe.workclass.replace(('Private','Self-emp-not-inc','Self-emp-inc','Federal-gov','Local-gov','State-gov','Without-pay','Never-worked'),(1,2,3,4,5,6,7,8), inplace=True)
    dataframe.education.replace(('Bachelors','Some-college','11th','HS-grad','Prof-school','Assoc-acdm','Assoc-voc','9th','7th-8th','12th','Masters','1st-4th','10th','Doctorate','5th-6th','Preschool'),(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16), inplace=True)
    dataframe.maritalstatus.replace(('Married-civ-spouse','Divorced','Never-married','Separated','Widowed','Married-spouse-absent','Married-AF-spouse'),(1,2,3,4,5,6,7), inplace=True)
    dataframe.occupation.replace(('Tech-support','Craft-repair','Other-service','Sales','Exec-managerial','Prof-specialty','Handlers-cleaners','Machine-op-inspct','Adm-clerical','Farming-fishing','Transport-moving','Priv-house-serv','Protective-serv','Armed-Forces'),(1,2,3,4,5,6,7,8,9,10,11,12,13,14), inplace=True)
    dataframe.relationship.replace(('Wife','Own-child','Husband','Not-in-family','Other-relative','Unmarried'),(1,2,3,4,5,6), inplace=True)
    dataframe.race.replace(('White','Asian-Pac-Islander','Amer-Indian-Eskimo','Other','Black'),(1,2,3,4,5), inplace=True)
    dataframe.gender.replace(('Female','Male'),(1,2), inplace=True)
    dataframe.nativecountry.replace(('United-States','Cambodia','England','Puerto-Rico','Canada','Germany','Outlying-US(Guam-USVI-etc)','India','Japan','Greece','South','China','Cuba','Iran','Honduras','Philippines','Italy','Poland','Jamaica','Vietnam','Mexico','Portugal','Ireland','France','Dominican-Republic','Laos','Ecuador','Taiwan','Haiti','Columbia','Hungary','Guatemala','Nicaragua','Scotland','Thailand','Yugoslavia','El-Salvador','Trinadad&Tobago','Peru','Hong','Holand-Netherlands'),(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41), inplace=True)
    dataframe.income.replace(('>50K','<=50K'),(1,2), inplace=True)
    dataframe.to_csv("clean_adultdata.csv", encoding='utf-8', index=False, header=False)

    return dataframe

def importData(file):
    set = csv.reader(open(file, "r"))
    data = list(set)
    for i in range(len(data)):
        data[i] = [float(x) for x in data[i]]
    return data


def cutData(data, ratio):
    size = int(len(data) * ratio)
    train = []
    data = list(data)
    while len(train) < size:
        index = random.randrange(len(data))
        train.append(data.pop(index))
    return [train, data]


def classSplit(data):
    splitted = {}
    for i in range(len(data)):
        vector = data[i]
        if (vector[-1] not in splitted):
            splitted[vector[-1]] = []
        splitted[vector[-1]].append(vector)
    return splitted


def mean(values):
    return sum(values) / float(len(values))


def stdev(values):
    average = mean(values)
    variance = sum([pow(x - average, 2) for x in values]) / float(len(values) - 1)
    return math.sqrt(variance)


def sumup(data):
    sums = [(mean(attribute), stdev(attribute)) for attribute in zip(*data)]
    del sums[-1]
    return sums


def sumsByClass(data):
    splitted = classSplit(data)
    sums = {}
    for value, instance in splitted.items():
        sums[value] = sumup(instance)
    return sums


def proba(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def probaByClass(sums, vector):
    probabilities = {}
    for value, classSums in sums.items():
        probabilities[value] = 1
        for i in range(len(classSums)):
            mean, stdev = classSums[i]
            x = vector[i]
            probabilities[value] *= proba(x, mean, stdev)
    return probabilities


def prediction(sums, vector):
    probabilities = probaByClass(sums, vector)
    largestAssociatedProba, largestProba = None, -1
    for value, probability in probabilities.items():
        if largestAssociatedProba is None or probability > largestProba:
            largestProba = probability
            largestAssociatedProba = value
    return largestAssociatedProba


def outputPredictions(sums, test):
    predictions = []
    for i in range(len(test)):
        output = prediction(sums, test[i])
        predictions.append(output)
    return predictions


def outputAccuracy(test, predictions):
    correct = 0
    for i in range(len(test)):
        if test[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(test))) * 100.0

def confusionMatrix(test, predictions):
    #TP = true positive, FP = false positive
    #TN = true negative, FN = false negative
    TP, FP, TN, FN = 0,0,0,0
    for i in range(len(test)):
        if test[i][-1] == 1:
            if test[i][-1] == predictions[i]:
                TP +=1
            else:
                FP +=1
        else:
            if test[i][-1] == predictions[i]:
                TN +=1
            else:
                FN +=1
    return[TP, FP, TN, FN]


def main():

    file='adult.csv'
    dataCleaning(file)
    cleaned_file = 'clean_adultdata.csv'
    x = [60, 70, 80]
    y=[]
    y1=[]
    y2=[]
    for i in range(len(x)):
        ratio = x[i]/100
        data = importData(cleaned_file)
        trainingSet, testSet = cutData(data, ratio)
        print("Split {0} rows into train={1} and test={2} rows".format(len(data), len(trainingSet), len(testSet)))
        # prepare model
        sums = sumsByClass(trainingSet)
        # test model
        predictions = outputPredictions(sums, testSet)
        accuracy = outputAccuracy(testSet, predictions)
        print('Accuracy: {0} %'.format(accuracy))
        y.append(accuracy)

        cm = confusionMatrix(testSet, predictions)
        TP = cm[0]
        FP = cm[1]
        TN = cm[2]
        FN = cm[3]
        #accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100
        recall = (TP / (TP + FN)) * 100
        y1.append(recall)
        print('Recall: ', recall, '%')

        precision = TP * 100 / (TP + FP)
        y2.append(precision)
        print('Precision: ',precision, '%')

        print("-------")

#Graphs
    #Y- axis: accuracy, x axis: hold out split
    plt.plot(x, y)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Accuracy')
    plt.plot(x, y, label="Accuracy")
    plt.show()

    #Y- axis: precision, x axis: hold out split
    plt.plot(x, y2)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Precision')
    plt.plot(x, y2, label="Precision")
    plt.show()

    #Y- axis: recall, x axis: hold out split
    plt.plot(x, y1)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Recall')
    plt.plot(x, y1, label="Recall")
    plt.show()


main()