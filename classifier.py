from sklearn.naive_bayes import GaussianNB
from os import listdir
import pandas as pd
import preprocess


gnb = GaussianNB()
attributes = []


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
    preprocess.numerate(df, attributes)
    makefit(df)

def makefit(df):
    x_train = df.drop(['class'], axis=1)
    y_train = df['class']
    gnb.fit(x_train, y_train)


def predict(binsnum, dataPath):
    for f in listdir(dataPath):
        if f == "test.csv":
            filename = dataPath + "/" + f
            test = pd.read_csv(filename)
            preprocess.clean(test, attributes)
            preprocess.discretisize(int(binsnum), test, attributes)
            preprocess.numerate(test, attributes)
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
