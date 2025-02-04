import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Read the dataset
print("Loading data...")
df = pd.read_csv('thyroid_cancer_risk_data.csv')

# Prepare features and target
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

# Create age groups
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 75, 100], 
                        labels=['<30', '30-45', '45-60', '60-75', '>75'])

# Prepare data for modeling
print("\nPreparing data for modeling...")
X = df[categorical_vars + numerical_vars + ['Risk_Score', 'Age_Group']].copy()
y = df['Diagnosis']

# Encode categorical variables
le = LabelEncoder()
for col in categorical_vars + ['Age_Group']:
    X[col] = le.fit_transform(X[col])
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
numerical_features = numerical_vars + ['Risk_Score']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'Neural Network': MLPClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

# Define hyperparameter grids for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'max_iter': [1000]
    },
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    },
    'LightGBM': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    },
    'Neural Network': {
        'hidden_layer_sizes': [(100,), (100, 50)],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01]
    },
    'AdaBoost': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1]
    }
}

# Train and evaluate models
print("\nTraining and evaluating models...")
best_models = {}
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Store best model
    best_models[name] = grid_search.best_estimator_
    
    # Make predictions
    y_pred = grid_search.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    results[name] = {
        'Best Parameters': grid_search.best_params_,
        'Best CV Score': grid_search.best_score_,
        'Test Classification Report': classification_report(y_test, y_pred),
        'ROC AUC': auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1])
    }

# Print results
print("\nModel Evaluation Results:")
print("=" * 50)
for name, result in results.items():
    print(f"\n{name}:")
    print("-" * 30)
    print(f"Best Parameters: {result['Best Parameters']}")
    print(f"Best CV Score: {result['Best CV Score']:.4f}")
    print(f"ROC AUC Score: {result['ROC AUC']:.4f}")
    print("\nClassification Report:")
    print(result['Test Classification Report'])

# Plot ROC curves
plt.figure(figsize=(10, 8))
for name, model in best_models.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend()
plt.savefig('model_comparison_roc.png')
plt.close()

# Feature importance analysis for tree-based models
tree_based_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']
feature_importance_df = pd.DataFrame()
feature_names = X_train.columns

for name in tree_based_models:
    if hasattr(best_models[name], 'feature_importances_'):
        feature_importance_df[name] = best_models[name].feature_importances_

feature_importance_df.index = feature_names

# Plot feature importance
plt.figure(figsize=(12, 8))
feature_importance_df.mean(axis=1).sort_values(ascending=True).plot(kind='barh')
plt.title('Average Feature Importance Across Tree-based Models')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Save best model
import joblib
best_model_name = max(results.items(), key=lambda x: x[1]['ROC AUC'])[0]
print(f"\nSaving best model ({best_model_name})...")
joblib.dump(best_models[best_model_name], 'best_thyroid_cancer_model.joblib')

# Save scaler for future use
joblib.dump(scaler, 'feature_scaler.joblib')

print("\nAnalysis complete! Results and visualizations have been saved.")
print(f"Best performing model: {best_model_name}")
print(f"Best ROC AUC Score: {results[best_model_name]['ROC AUC']:.4f}")
