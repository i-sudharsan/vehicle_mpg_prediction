## mpg_prediction.py

#Step1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score


#Step2: Load Dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url)

print("Dataset Shape :", df.shape)
print(df.head())
print(df.columns)


# Step3: Handle missing values
from sklearn.impute import SimpleImputer

#Check Missing values
print(df.isnull().sum())

#Replace missing values with median # horsepower values
imputer = SimpleImputer(strategy='median')
df['horsepower'] = imputer.fit_transform(df[['horsepower']])    

print("\nMissing values after imputation:")
print(df.isnull().sum())


# Step4: Identify numeric and categorical features
df = df.drop(columns=['name','origin']) # Drop non-numeric columns

X = df[['horsepower','weight','acceleration','displacement','cylinders','model_year']]
y = df['mpg']

# Step5: Train-Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# Step6: Scale Features
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Create a pipeline for numeric features
numeric_pipeline = Pipeline(steps=[
    ('scaler',StandardScaler())])

#Fit & Transform train & test data
X_train_scaled = numeric_pipeline.fit_transform(X_train)
X_test_scaled = numeric_pipeline.transform(X_test)

print("Before Scaling:\n", X_train.head())
print(X_train.describe().T[['mean','std']])
print("\nAfter Scaling:\n", X_train_scaled[:5])
print("X_train.scaled.mean():", X_train_scaled.mean(axis=0) )
print("X_train.scaled.std():", X_train_scaled.std(axis=0) ) 


# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Step7: Train Model
model = LinearRegression()
model.fit(X_train_scaled,y_train)
print("Model trained successfully.")

# Step8: Predictions
y_pred = model.predict(X_test_scaled)
print("Predictions on test set completed.")

# Step9: Evaluate
# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

# Step10: Visualization

# # Step 7: Visualize predictions vs actual MPG
# import matplotlib.pyplot as plt

# plt.figure(figsize=(7,5))
# plt.scatter(y_test, y_pred, alpha=0.7)
# plt.xlabel("Actual MPG")
# plt.ylabel("Predicted MPG")
# plt.title("Actual vs Predicted MPG")
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.show()

# Using seaborn for better visualization
# plt.figure(figsize=(6,6))
# sns.scatterplot(x=y_test,y=y_pred)
# plt.xlabel('Actual MPG')
# plt.ylabel('Predicted MPG')
# plt.title('Actual vs Predicted MPG')
# plt.show()

#Step11A: Try Ridge Regression (adds Regularization)
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled,y_train)
y_pred_ridge = ridge.predict(X_test_scaled )
print("Ridge Regression Predictions on test set completed.")

r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f"\nRidge Regression Model Evaluation:")
print(f"MAE  : {mae_ridge:.2f}")
print(f"RMSE : {rmse_ridge:.2f}")
print(f"R²   : {r2_ridge:.2f}")

#Step11B: Try Random Forest Regression (non-linear model)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest Predictions on test set completed.")   

r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"\nRandom Forest Model Evaluation:")

print(f"MAE  : {mae_rf:.2f}")
print(f"RMSE : {rmse_rf:.2f}")
print(f"R²   : {r2_rf:.2f}")

# Step12: Compare Models Visually
import matplotlib.pyplot as plt

#Collect all model results
results = {
    'Linear Regression': (mae, rmse, r2),
    'Ridge Regression': (mae_ridge, rmse_ridge, r2_ridge),
    'Random Forest': (mae_rf, rmse_rf, r2_rf)
}

#Create DataFrame for visualization
results_df = pd.DataFrame(results, index=['MAE', 'RMSE', 'R²']).T
print("\nModel Comparison:\n", results_df)

# #Plot R2 Comparison
# plt.figure(figsize=(8,5))
# plt.bar(results_df.index, results_df['R²'], color=['blue','orange','green'])
# plt.ylabel('R² Score')

# plt.title('Model R² Comparison')
# plt.ylim(0,1)
# plt.show()

# Step13: Save Trained Model
import joblib

#save both model and column names for future use
joblib.dump(rf,'mpg_random_forest_model.pkl')
joblib.dump(list(X.columns),'mpg_model_features.pkl')

print("Model and feature names saved successfully.")

# Step14: Load model & predict for new data
model = joblib.load("mpg_random_forest_model.pkl")
feature_names = joblib.load("mpg_model_features.pkl")

#New data for prediction
new_car = pd.DataFrame({
    'horsepower':[130],
    'weight':[3504],
    'acceleration':[12.0],
    'displacement':[307],
    'cylinders':[8],
    'model_year':[70]
})[feature_names]

predicted_mpg = model.predict(new_car)
print(f"\nPredicted MPG for new car data: {predicted_mpg[0]:.2f}")

