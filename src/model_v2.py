import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the preprocessed data
df = pd.read_csv('../data/preprocessed_data.csv')
df = df.drop('country', axis=1)

X = df.drop('rating', axis=1)
y = df['rating'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
xgb_model.fit(X_train, y_train)


# Predict the ratings
y_pred = xgb_model.predict(X_test)



# Performance metrics
# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean squared error: {mse}')
print(f'Root mean squared error: {rmse}')
print(f'Mean absolute error: {mae}')
print(f'R2 score: {r2}')

# Plot the feature importances
xgb.plot_importance(xgb_model)
plt.title('Feature importances')
plt.show()







