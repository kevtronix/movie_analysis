import pandas as pd
import glob
import warnings

path = '../data/seperate_data/'
all_files = glob.glob(path + "/*.csv")

all_dataframes = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    all_dataframes.append(df)


combined_df = pd.concat(all_dataframes, axis=0, ignore_index=True)

combined_df.to_csv('../data/combined_movie_data.csv', index=False)