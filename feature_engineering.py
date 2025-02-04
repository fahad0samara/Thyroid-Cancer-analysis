import pandas as pd
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
