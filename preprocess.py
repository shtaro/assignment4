import pandas as pd
import numpy as np
from scipy.stats import mode


def discretisize(bins, df, attributes):
    for att in attributes:
        if att[1] == 'NUMERIC':
            min = df[att[0]].min()
            max = df[att[0]].max()
            w = (max-min)/bins #calculate the width of the bins
            cut_points = []
            for x in range(1, bins):
                cut_points.append(min+(x*w))
            cut_points = [min] + cut_points + [max] #set the cut point for the bins
            labels = range(1, len(cut_points))
            df[att[0]] = pd.cut(df[att[0]], bins=cut_points, labels=labels, include_lowest=True)


def clean(df, attributes):
    for att in attributes:
        if att[1] == 'NUMERIC':
            #get the mean for each class for this attribute
            yesMean = df.loc[(df['class'] == 'Y')][att[0]].mean()
            noMean = df.loc[(df['class'] == 'N')][att[0]].mean()
            #complete the missing value according to the class
            df.loc[(df['class'] == 'Y') & (np.isnan(df[att[0]])), att[0]] = yesMean
            df.loc[(df['class'] == 'N') & (np.isnan(df[att[0]])), att[0]] = noMean
        else:
            df[att[0]].fillna(mode(df[att[0]]).mode[0], inplace=True) #complete the missing value according to the most common value
