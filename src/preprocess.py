import pandas as pd
import numpy as np



# Helper functions 
def run_length_to_minutes(run_length):
    hours = 0
    minutes = 0
    if 'h' in run_length:
        hours = int(run_length.split('h')[0])
    if 'min' in run_length:
        minutes = int(run_length.split('min')[0].split()[-1])
    return hours * 60 + minutes

# Load the data 
df = pd.read_csv('../data/combined_movie_data.csv')
print(df.head())

# Drop all rows with duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Drop the review_url 
df.drop('review_url', axis=1, inplace=True)

# Drop the name of the movie
df.drop('name', axis=1, inplace=True)

# Ensure 'year' is an integer and 'run_length' is in minutes
df['year'] = df['year'].astype(int)
df['run_length'] = df['run_length'].apply(run_length_to_minutes)


# One-hot encode genres and drop the column
genres = df['genres'].str.get_dummies('; ')
df = df.join(genres)
df.drop('genres', axis=1, inplace=True)
print(df.head())


# Extract the release year and country from the 'release_date' and drop the column
df['release_year'] = pd.to_datetime(df['release_date'].str.extract(r'(\d{4})')[0], format='%Y').dt.year
df['country'] = df['release_date'].str.extract(r'\((.*?)\)')
df.drop('release_date', axis=1, inplace=True)
print(df.head())

# One-hot encode the country and drop the column
countries = df['country'].str.get_dummies(', ')
df = df.join(countries)
print(df.head())

''' 
After analysis it appears that movie_rated doesn't have to be applied and can instead be dropped

# Check for missing values and drop the rows with missing values
print(df['movie_rated'].isnull().sum())
print(df['movie_rated'].str.contains('null', case=False, na=False).sum())

df['movie_rated'].replace('null', np.nan, inplace=True, regex=True)
df['movie_rated'].fillna('Unknown', inplace=True)

# One-hot encode 'movie_rated' and drop the column
rated_dummies = pd.get_dummies(df['movie_rated'], prefix='rated', dummy_na=False)
df = pd.concat([df, rated_dummies], axis=1)
df.drop('movie_rated', axis=1, inplace=True)
print(df.head())

'''
df.drop('movie_rated', axis=1, inplace=True)

# Ensure the rating to a float 
df['rating'] = df['rating'].astype(float)

# Ensure 'num_raters' and 'num_reviews' are integers
df['num_raters'] = df['num_raters'].astype(int)
df['num_reviews'] = df['num_reviews'].astype(int)
print(df.head())


# Save the preprocessed data
df.to_csv('../data/preprocessed_data.csv', index=False)




