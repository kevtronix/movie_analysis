import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the preprocessed data
df = pd.read_csv('../data/preprocessed_data.csv')
df = df.drop('country', axis=1)

X = df.drop('rating', axis=1)
y = df['rating'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Get the feature importances
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)



# Plot the feature importances
plt.figure(figsize=(12, 6))
feature_importances.sort_values().plot(kind='barh')
plt.title('Feature importances')
plt.show()

