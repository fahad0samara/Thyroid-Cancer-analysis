import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import joblib

# Load data
print("Loading data...")
df = pd.read_csv('thyroid_cancer_risk_data.csv')

# Define features
categorical_vars = ['Gender', 'Country', 'Ethnicity', 'Family_History', 
                   'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 
                   'Obesity', 'Diabetes']
numerical_vars = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']

# Feature Engineering
print("\nPerforming feature engineering...")
# Create risk score
df['Risk_Score'] = (
    (df['Family_History'] == 'Yes').astype(int) * 2 +
    (df['Radiation_Exposure'] == 'Yes').astype(int) * 1.5 +
    (df['Iodine_Deficiency'] == 'Yes').astype(int) * 1.5 +
    (df['Smoking'] == 'Yes').astype(int) +
    (df['Obesity'] == 'Yes').astype(int) +
    (df['Diabetes'] == 'Yes').astype(int)
)

# Create hormone ratios
df['TSH_T3_Ratio'] = df['TSH_Level'] / df['T3_Level']
df['TSH_T4_Ratio'] = df['TSH_Level'] / df['T4_Level']
df['T3_T4_Ratio'] = df['T3_Level'] / df['T4_Level']

# Create age groups
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 75, 100], 
                        labels=['<30', '30-45', '45-60', '60-75', '>75'])

# Create interaction features
df['Age_TSH'] = df['Age'] * df['TSH_Level']
df['Nodule_TSH'] = df['Nodule_Size'] * df['TSH_Level']

# Update feature lists with exact order
categorical_vars = ['Gender', 'Country', 'Ethnicity', 'Family_History', 
                   'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 
                   'Obesity', 'Diabetes', 'Age_Group']

numerical_vars = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size',
                 'Risk_Score', 'TSH_T3_Ratio', 'TSH_T4_Ratio', 'T3_T4_Ratio',
                 'Age_TSH', 'Nodule_TSH']

# Prepare features for modeling
print("\nPreparing data for modeling...")
X = df[categorical_vars + numerical_vars].copy()
y = (df['Diagnosis'] == 'Malignant').astype(int)

# Encode categorical variables
le = LabelEncoder()
for col in categorical_vars:
    X[col] = le.fit_transform(X[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_vars] = scaler.fit_transform(X_train[numerical_vars])
X_test[numerical_vars] = scaler.transform(X_test[numerical_vars])

# Train XGBoost model
print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Save model and preprocessors
print("\nSaving model and preprocessors...")
joblib.dump(model, 'best_thyroid_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("\nDone! Model and preprocessors have been saved.")

# Print model performance
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report
print("\nModel Performance:")
print(classification_report(y_test, y_pred))
