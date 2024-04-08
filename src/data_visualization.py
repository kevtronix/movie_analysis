import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Visualization

# Load the preprocessed data
df = pd.read_csv('../data/preprocessed_data.csv')

# Ratings distribution
sns.histplot(data=df, x='rating')
plt.show()



# Genres distribution
genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 
          'Drama', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 
          'Sci-Fi', 'Sport', 'Thriller', 'War' ]

for genre in genres:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=genre, y='rating', data=df)
    plt.title(f'Rating distribution for {genre} movies')
    plt.show()


# Release year distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='release_year')
plt.title('Release year distribution')
plt.show()


# Country distribution
plt.figure(figsize=(12, 6))
sns.countplot(y='country', data=df, order=df['country'].value_counts().index)
plt.title('Movie count by country')
plt.show()
