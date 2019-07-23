import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder


def numerate(df, attributes):
    le = LabelEncoder()
    for att in attributes:
        if att[1] != 'NUMERIC':
            df[att[0]] = le.fit_transform(df[att[0]])


def discretisize(bins, df, attributes):
    for att in attributes:
        if att[1] == 'NUMERIC':
            min = df[att[0]].min()
            max = df[att[0]].max()
            w = (max-min)/bins
            cut_points = []
            for x in range(1, bins):
                cut_points.append(min+(x*w))
            cut_points = [min] + cut_points + [max]
            labels = range(1, len(cut_points))
            df[att[0]] = pd.cut(df[att[0]], bins=cut_points, labels=labels, include_lowest=True)


def clean(df, attributes):
    for att in attributes:
        if att[1] == 'NUMERIC':
            yesMean= df.loc[(df['class'] == 'Y')][att[0]].mean()
            noMean= df.loc[(df['class'] == 'N')][att[0]].mean()

            df.loc[(df['class'] == 'Y') & (np.isnan(df[att[0]])), att[0]] = yesMean
            df.loc[(df['class'] == 'N') & (np.isnan(df[att[0]])), att[0]] = noMean

        else:
            df[att[0]].fillna(mode(df[att[0]]).mode[0], inplace=True)