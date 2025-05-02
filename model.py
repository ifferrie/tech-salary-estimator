import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv('salaries_clean.csv')
df = df[['job_title', 'total_experience_years', 'job_title_rank', 'location_state', 'annual_base_pay']]

# Clean the dataset
df.dropna(inplace=True)


df_encoded = pd.get_dummies(df, columns=['job_title', 'job_title_rank', 'location_state'])
X = df_encoded.drop('annual_base_pay', axis=1)
y = df_encoded['annual_base_pay']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and feature columns
joblib.dump(model, 'salary_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')

# evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")