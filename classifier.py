from os import listdir
import pandas as pd
import preprocess


attributes = []
probsClass = {}
probs = {}

def buildModel(binsNum, dataPath):
    #open the Structure and train files
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
    preprocess.clean(df, attributes) #complete the missing values
    preprocess.discretisize(int(binsNum), df, attributes) #discretisize the data
    makefit(df, binsNum) #build the model

def makefit(df, binsNum):
    m = 2
    #calculate the probability of each class
    numY = len(df.loc[(df['class'] == 'Y')]) * 1.0
    numN = len(df.loc[(df['class'] == 'N')]) * 1.0
    n = len(df.index)
    probY = numY / n
    probN = numN / n
    probsClass["Y"] = probY
    probsClass["N"] = probN
    for att in attributes:
        if att[1] == 'NUMERIC':
            vals = range(1, int(binsNum)+1)
            p = 1 / (int(binsNum)*1.0)
            valDic = {}
            for val in vals:
                # calculate the probability of each class given this value
                numRecY = len(df.loc[(df['class'] == 'Y') & (df[att[0]] == val)]) * 1.0
                numRecN = len(df.loc[(df['class'] == 'N') & (df[att[0]] == val)]) * 1.0
                prob1 = (numRecY + m * p) / (numY + m)
                prob2 = (numRecN + m * p) / (numN + m)
                probDic = {}
                probDic['Y'] = prob1
                probDic['N'] = prob2
                valDic[val] = probDic
            probs[att[0]] = valDic
        else:
            vals = att[1].split(",")
            p = 1 / (len(vals)*1.0)
            valDic = {}
            for val in vals:
                # calculate the probability of each class given this value
                numRecY = len(df.loc[(df['class'] == 'Y') & (df[att[0]] == val)]) * 1.0
                numRecN = len(df.loc[(df['class'] == 'N') & (df[att[0]] == val)]) * 1.0
                prob1 = (numRecY + m * p) / (numY + m)
                prob2 = (numRecN + m * p) / (numN + m)
                probDic = {}
                probDic['Y'] = prob1
                probDic['N'] = prob2
                valDic[val] = probDic
            probs[att[0]] = valDic


pred = []


def predict(binsnum, dataPath):
    for f in listdir(dataPath):
        #read the test file
        if f == "test.csv":
            filename = dataPath + "/" + f
            test = pd.read_csv(filename)
            preprocess.clean(test, attributes)
            preprocess.discretisize(int(binsnum), test, attributes)
            test2 = test.drop(['class'], axis=1)
            for index, row in test2.iterrows():
                calcprobY = 1
                calcprobN = 1
                # calculate the probability of each class given this row
                for att in attributes:
                    if att[0] != 'class':
                        calcprobY = calcprobY * (((probs[att[0]])[row[att[0]]])['Y'])
                        calcprobN = calcprobN * (probs[att[0]][row[att[0]]]['N'])
                probY = probsClass['Y'] * calcprobY
                probN = probsClass['N'] * calcprobN
                #choose the class according to the higher probability
                if probY > probN:
                    pred.append('yes')
                else:
                    pred.append('no')
            file = open(dataPath+"/output.txt", "w")
            j = 0
            for i in range(1, len(pred)+1):
                file.write(str(i) + " " + pred[j]+"\n")
                j += 1
            file.close()
