import pandas as pd
import numpy as np
import os
import string

def read_data() -> pd.DataFrame:
    directory = './testlanguageID/'
    letters = [char for char in string.ascii_lowercase]
    letters.append('space')
    letters.append('type')
    df = pd.DataFrame(columns=letters)

    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        freq={}
        # initialize a,z + <SPACE> as 0 freq initially
        freq = dict.fromkeys(letters,0) 
        with open(file) as fileObj:
            for line in fileObj:  
                for ch in line:
                    if ch == '\n':
                        continue
                    elif ch == ' ':
                        freq['space'] += 1
                    else:
                        freq[ch] += 1
            freq['type'] = filename[0]
        
        df_dictionary = pd.DataFrame([freq])
        df = pd.concat([df, df_dictionary], ignore_index=True)
        print(df)

    return df
if __name__ ==  '__main__':
    df = read_data()
    #priors = calculate_prior()

