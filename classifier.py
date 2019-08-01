from sklearn.naive_bayes import GaussianNB
from os import listdir
import pandas as pd
import preprocess


gnb = GaussianNB()
attributes = []
probs = []
probsClass = []

def buildModel(binsNum, dataPath):

    for f in listdir(dataPath):
        if f == "Structure.txt":
            filename = dataPath+"/"+f
            file = open(filename, 'r')
            for line in file:
                name = line.split(" ")[1]
                if line.split(" ")[2][0] == 'N':
                    values = 'NUMERIC'
                else:
                    values = line.split("{")[1]
                    values = values.replace("}", "")
                    values = values.replace("\n", "")
                o = [name, values]
                attributes.append(o)
        if f == "train.csv":
            filename = dataPath+"/"+f
            df = pd.read_csv(filename)
    preprocess.clean(df, attributes)
    preprocess.discretisize(int(binsNum), df, attributes)
    #preprocess.numerate(df, attributes)
    makefit(df, binsNum)

def makefit(df, binsNum):
    m = 2
    numY = len(df.loc[(df['class'] == 'Y')]) * 1.0
    numN = len(df.loc[(df['class'] == 'N')]) * 1.0
    n = len(df.index)
    probY = numY / n
    probN = numN / n
    probsClass.append(['Y', probY])
    probsClass.append(['N', probN])
    for att in attributes:
        if att[1] == 'NUMERIC':
            vals = range(1, int(binsNum)+1)
            p = 1 / (int(binsNum)*1.0)
            for val in vals:
                numRecY = len(df.loc[(df['class'] == 'Y') & (df[att[0]] == val)]) * 1.0
                numRecN = len(df.loc[(df['class'] == 'N') & (df[att[0]] == val)]) * 1.0
                prob1 = (numRecY + m * p) / (numY + m)
                prob2 = (numRecN + m * p) / (numN + m)
                o1 = (att[0], val, 'Y', prob1)
                o2 = (att[0], val, 'N', prob2)
                probs.append(o1)
                probs.append(o2)
        else:
            vals = att[1].split(",")
            p = 1 / (len(vals)*1.0)
            for val in vals:
                numRecY = len(df.loc[(df['class'] == 'Y') & (df[att[0]] == val)]) * 1.0
                numRecN = len(df.loc[(df['class'] == 'N') & (df[att[0]] == val)]) * 1.0
                prob1 = (numRecY + m * p) / (numY + m)
                prob2 = (numRecN + m * p) / (numN + m)
                o1 = (att[0], val, 'Y', prob1)
                o2 = (att[0], val, 'N', prob2)
                probs.append(o1)
                probs.append(o2)


def predict(binsnum, dataPath):
    for f in listdir(dataPath):
        if f == "test.csv":
            filename = dataPath + "/" + f
            test = pd.read_csv(filename)
            preprocess.clean(test, attributes)
            preprocess.discretisize(int(binsnum), test, attributes)
            #preprocess.numerate(test, attributes)
            test2 = test.drop(['class'], axis=1)
            pred = gnb.predict(test2)
            pred2 = []
            for j in range(0, len(pred)):
                if pred[j] == 1:
                    pred2.append("yes")
                else:
                    pred2.append("no")
            file = open(dataPath+"/output.txt", "w")
            j = 0
            for i in range(1, len(pred)+1):
                file.write(str(i) + " " + pred2[j]+"\n")
                j += 1
            file.close()
