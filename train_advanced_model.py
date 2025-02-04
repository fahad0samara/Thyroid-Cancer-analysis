import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
import optuna
import joblib
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    """Create all features consistently for both training and prediction"""
    # Basic features
    df['Risk_Score'] = (
        (df['Family_History'] == 'Yes').astype(int) * 2 +
        (df['Radiation_Exposure'] == 'Yes').astype(int) * 1.5 +
        (df['Iodine_Deficiency'] == 'Yes').astype(int) * 1.5 +
        (df['Smoking'] == 'Yes').astype(int) +
        (df['Obesity'] == 'Yes').astype(int) +
        (df['Diabetes'] == 'Yes').astype(int)
    )
    
    # Hormone ratios
    df['TSH_T3_Ratio'] = df['TSH_Level'] / df['T3_Level']
    df['TSH_T4_Ratio'] = df['TSH_Level'] / df['T4_Level']
    df['T3_T4_Ratio'] = df['T3_Level'] / df['T4_Level']
    
    # Age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 75, 100],
                            labels=['<30', '30-45', '45-60', '60-75', '>75'])
    
    # Interactions
    df['Age_TSH'] = df['Age'] * df['TSH_Level']
    df['Nodule_TSH'] = df['Nodule_Size'] * df['TSH_Level']
    
    # Hormone balance score
    df['Hormone_Balance'] = np.abs(
        (df['TSH_Level'] - df['TSH_Level'].mean()) / df['TSH_Level'].std() +
        (df['T3_Level'] - df['T3_Level'].mean()) / df['T3_Level'].std() +
        (df['T4_Level'] - df['T4_Level'].mean()) / df['T4_Level'].std()
    )
    
    # Clinical severity score
    df['Clinical_Severity'] = (
        (df['TSH_Level'] > df['TSH_Level'].quantile(0.75)).astype(int) +
        (df['Nodule_Size'] > df['Nodule_Size'].quantile(0.75)).astype(int) +
        (df['Age'] > df['Age'].quantile(0.75)).astype(int)
    )
    
    # Risk factor combinations
    risk_factors = ['Family_History', 'Radiation_Exposure', 'Smoking', 'Obesity']
    for i in range(len(risk_factors)):
        for j in range(i+1, len(risk_factors)):
            col_name = f"{risk_factors[i]}_{risk_factors[j]}_Combo"
            df[col_name] = ((df[risk_factors[i]] == 'Yes') & 
                           (df[risk_factors[j]] == 'Yes')).astype(int)
    
    # Polynomial features for numerical variables
    num_vars = ['TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[num_vars])
    poly_features = pd.DataFrame(
        poly_features[:, len(num_vars):],
        columns=[f'Poly_{i}' for i in range(poly_features.shape[1]-len(num_vars))]
    )
    df = pd.concat([df, poly_features], axis=1)
    
    return df

def get_feature_names():
    """Get the exact list of features in the correct order"""
    categorical_vars = ['Gender', 'Country', 'Ethnicity', 'Family_History',
                       'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking',
                       'Obesity', 'Diabetes', 'Age_Group']
    
    numerical_base = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
    derived_features = ['Risk_Score', 'TSH_T3_Ratio', 'TSH_T4_Ratio', 'T3_T4_Ratio',
                       'Age_TSH', 'Nodule_TSH', 'Hormone_Balance', 'Clinical_Severity']
    
    # Risk factor combinations
    risk_factors = ['Family_History', 'Radiation_Exposure', 'Smoking', 'Obesity']
    combo_features = []
    for i in range(len(risk_factors)):
        for j in range(i+1, len(risk_factors)):
            combo_features.append(f"{risk_factors[i]}_{risk_factors[j]}_Combo")
    
    # Polynomial features
    num_vars = ['TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
    n_poly_features = int(len(num_vars) * (len(num_vars) + 1) / 2)
    poly_features = [f'Poly_{i}' for i in range(n_poly_features)]
    
    all_features = (categorical_vars + numerical_base + derived_features + 
                   combo_features + poly_features)
    
    return all_features

print("Loading and preparing data...")

# Load data
df = pd.read_csv('thyroid_cancer_risk_data.csv')

# Create features
df = create_features(df)

# Get feature names
feature_names = get_feature_names()

# Prepare target
y = (df['Diagnosis'] == 'Malignant').astype(int)

# Prepare features
X = df[feature_names].copy()

# Encode categorical variables
categorical_vars = ['Gender', 'Country', 'Ethnicity', 'Family_History',
                   'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking',
                   'Obesity', 'Diabetes', 'Age_Group']

le = LabelEncoder()
for col in categorical_vars:
    X[col] = le.fit_transform(X[col])

# Scale numerical features
numerical_features = [col for col in feature_names if col not in categorical_vars]
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Feature selection using XGBoost
print("\nPerforming feature selection...")
base_model = XGBClassifier(random_state=42)
selector = SelectFromModel(base_model, prefit=False)
selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.get_support()].tolist()

X_train = X_train[selected_features]
X_val = X_val[selected_features]
X_test = X_test[selected_features]

# Optuna optimization for XGBoost
def optimize_xgb(trial, X_train, y_train, X_val, y_val):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0)
    }
    
    model = XGBClassifier(**param, random_state=42)
    model.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             early_stopping_rounds=50,
             verbose=False)
    
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

# Optimize XGBoost
print("\nOptimizing XGBoost hyperparameters...")
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: optimize_xgb(trial, X_train, y_train, X_val, y_val),
              n_trials=50, show_progress_bar=True)

# Train final models
print("\nTraining final models...")

# XGBoost
xgb = XGBClassifier(**study.best_params, random_state=42)
xgb.fit(X_train, y_train)

# LightGBM
lgb = LGBMClassifier(random_state=42)
lgb.fit(X_train, y_train)

# CatBoost
cat = CatBoostClassifier(random_state=42, verbose=False)
cat.fit(X_train, y_train)

# Create ensemble
estimators = [
    ('xgb', xgb),
    ('lgb', lgb),
    ('cat', cat)
]

voting = VotingClassifier(estimators=estimators, voting='soft')
voting.fit(X_train, y_train)

# Evaluate models
print("\nEvaluating models...")
models = {
    'XGBoost': xgb,
    'LightGBM': lgb,
    'CatBoost': cat,
    'Voting Ensemble': voting
}

best_auc = 0
best_model = None

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n{name} Results:")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if auc > best_auc:
        best_auc = auc
        best_model = model

# Save best model and preprocessors
print("\nSaving best model and preprocessors...")
joblib.dump(best_model, 'best_thyroid_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'label_encoder.joblib')
joblib.dump(selected_features, 'selected_features.joblib')
joblib.dump(feature_names, 'feature_names.joblib')

# Save feature creation function
with open('feature_engineering.py', 'w') as f:
    f.write("""import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def create_features(df):
    # Basic features
    df['Risk_Score'] = (
        (df['Family_History'] == 'Yes').astype(int) * 2 +
        (df['Radiation_Exposure'] == 'Yes').astype(int) * 1.5 +
        (df['Iodine_Deficiency'] == 'Yes').astype(int) * 1.5 +
        (df['Smoking'] == 'Yes').astype(int) +
        (df['Obesity'] == 'Yes').astype(int) +
        (df['Diabetes'] == 'Yes').astype(int)
    )
    
    # Hormone ratios
    df['TSH_T3_Ratio'] = df['TSH_Level'] / df['T3_Level']
    df['TSH_T4_Ratio'] = df['TSH_Level'] / df['T4_Level']
    df['T3_T4_Ratio'] = df['T3_Level'] / df['T4_Level']
    
    # Age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 75, 100],
                            labels=['<30', '30-45', '45-60', '60-75', '>75'])
    
    # Interactions
    df['Age_TSH'] = df['Age'] * df['TSH_Level']
    df['Nodule_TSH'] = df['Nodule_Size'] * df['TSH_Level']
    
    # Hormone balance score
    df['Hormone_Balance'] = np.abs(
        (df['TSH_Level'] - 2.5) / 2.0 +
        (df['T3_Level'] - 1.8) / 0.5 +
        (df['T4_Level'] - 8.0) / 2.0
    )
    
    # Clinical severity score
    df['Clinical_Severity'] = (
        (df['TSH_Level'] > 4.0).astype(int) +
        (df['Nodule_Size'] > 2.0).astype(int) +
        (df['Age'] > 60).astype(int)
    )
    
    # Risk factor combinations
    risk_factors = ['Family_History', 'Radiation_Exposure', 'Smoking', 'Obesity']
    for i in range(len(risk_factors)):
        for j in range(i+1, len(risk_factors)):
            col_name = f"{risk_factors[i]}_{risk_factors[j]}_Combo"
            df[col_name] = ((df[risk_factors[i]] == 'Yes') & 
                           (df[risk_factors[j]] == 'Yes')).astype(int)
    
    # Polynomial features for numerical variables
    num_vars = ['TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[num_vars])
    poly_features = pd.DataFrame(
        poly_features[:, len(num_vars):],
        columns=[f'Poly_{i}' for i in range(poly_features.shape[1]-len(num_vars))]
    )
    df = pd.concat([df, poly_features], axis=1)
    
    return df
""")

print(f"\nBest model saved with ROC AUC: {best_auc:.4f}")
print("Training complete!")
