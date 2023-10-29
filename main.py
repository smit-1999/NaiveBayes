import pandas as pd
import numpy as np
import os
import string

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

    print('Prior values for e, j ,s with laplace factor are',
          e_prior, j_prior, s_prior)
    return e_prior, j_prior, s_prior


if __name__ == '__main__':
    directory = './languageID/'
    df = read_data(directory)
    print('Loaded dataframe shape is', df.shape)
    train = df.loc[df['file_name'].astype(str).map(len) == 6]
    test = df.loc[df['file_name'].astype(str).map(len) != 6]

    priors = calculate_prior(train)
