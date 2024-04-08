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

y_pred = model.predict(X_test)

# Determine prediction metrics
mae = mean_absolute_error(y_test, y_pred) 
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Get the feature importances
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)



# Plot the feature importances
plt.figure(figsize=(20, 10))
feature_importances.sort_values().plot(kind='barh')
plt.title('Feature importances')
plt.show()

