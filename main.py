import pandas as pd
import numpy as np
import string
import math
import os
# reads all files from the directory and loads all entries in dataframe
# returns the dataframe


def read_data(directory: string) -> pd.DataFrame:
    letters = [char for char in string.ascii_lowercase]
    letters.append('space')
    # y label: e, s or j
    letters.append('type')
    features = letters
    features.append('file_name')
    df = pd.DataFrame(columns=features)

    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        feature = {}
        # initialize a,z + <SPACE> as 0 freq initially
        feature = dict.fromkeys(letters, 0)
        with open(file) as fileObj:
            for line in fileObj:
                for ch in line:
                    if ch == '\n':
                        continue
                    elif ch == ' ':
                        feature['space'] += 1
                    else:
                        feature[ch] += 1
            feature['type'] = filename[0]
        feature['file_name'] = filename
        df_dictionary = pd.DataFrame([feature])
        df = pd.concat([df, df_dictionary], ignore_index=True)

    return df


def calculate_prior(df: pd.DataFrame):
    # laplace smoothing factor
    alpha = 0.5
    total = len(train) + 3*alpha
    e_prior = (
        len(df.loc[df['file_name'].astype(str).str[0] == 'e']) + alpha) / total
    s_prior = (
        len(df.loc[df['file_name'].astype(str).str[0] == 's']) + alpha)/total
    j_prior = (
        len(df.loc[df['file_name'].astype(str).str[0] == 'j']) + alpha)/total

    print('Prior probability values for e, j ,s with laplace factor are',
          e_prior, j_prior, s_prior)
    return math.log(e_prior), math.log(j_prior), math.log(s_prior)


def calculate_class_conditional(df: pd.DataFrame, label: string):
    # laplace smoothing factor
    alpha = 0.5

    # in whole training dataframe filter out rows for the particular label
    df = df.loc[df['file_name'].astype(str).str[0] == label]

    # print('Dataframe having filtered out rows for the particular label', df)

    # take only characters
    df = df.iloc[:, : 27]
    # print('Dataframe after filtering only letters and spaces as the features', df)

    likelihood = {}
    likelihood = dict.fromkeys(df.columns, 0)
    total = df.to_numpy().sum() + (alpha*27)
    print('Total sum of dataframe along with laplace smoothing for label',
          label, 'is',  total)

    for col in df.columns:
        col_sum = df[col].to_numpy().sum() + alpha
        prob = col_sum/total
        log_prob = math.log(prob)
        likelihood[col] = (col_sum, prob, log_prob)
    print('Likelihood dictionary for label', label, 'is', likelihood)

    # sum of probabilities should be 1
    res = tuple(sum(x) for x in zip(*likelihood.values()))
    print('Sum of tuple values of likelihood dictionary for label', label, 'is',  res)
    return likelihood


if __name__ == '__main__':
    directory = './languageID/'
    df = read_data(directory)
    print('Loaded dataframe shape is', df.shape)

    train = df.loc[df['file_name'].astype(str).map(len) == 6]
    test = df.loc[df['file_name'].astype(str).map(len) != 6]

    priors = calculate_prior(train)
    e_conditional = calculate_class_conditional(train, 'e')
    j_conditional = calculate_class_conditional(train, 'j')
    s_conditional = calculate_class_conditional(train, 's')
