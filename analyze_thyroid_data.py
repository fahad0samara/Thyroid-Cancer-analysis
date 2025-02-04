import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]

# Read the dataset
df = pd.read_csv('thyroid_cancer_risk_data.csv')

# Create directories for different types of plots
for dir_name in ['distributions', 'relationships', 'categorical', 'clinical', 'advanced']:
    import os
    os.makedirs(f'plots/{dir_name}', exist_ok=True)

# 1. Distribution Analysis for Numerical Variables
numerical_vars = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']

# Create distribution plots with multiple components
for var in numerical_vars:
    plt.figure(figsize=(15, 10))
    
    # Main distribution plot
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x=var, hue='Diagnosis', multiple="stack", bins=30)
    plt.title(f'Distribution of {var} by Diagnosis')
    
    # Box plot
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, y=var, x='Diagnosis')
    plt.title(f'Box Plot of {var} by Diagnosis')
    
    # Violin plot
    plt.subplot(2, 2, 3)
    sns.violinplot(data=df, y=var, x='Diagnosis')
    plt.title(f'Violin Plot of {var} by Diagnosis')
    
    # KDE plot
    plt.subplot(2, 2, 4)
    sns.kdeplot(data=df, x=var, hue='Diagnosis', fill=True)
    plt.title(f'KDE Plot of {var} by Diagnosis')
    
    plt.tight_layout()
    plt.savefig(f'plots/distributions/{var}_analysis.png')
    plt.close()

# 2. Clinical Measurements Relationships
# Create pair plot for clinical measurements
clinical_vars = ['TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
plt.figure(figsize=(20, 20))
sns.pairplot(df[clinical_vars + ['Diagnosis']], hue='Diagnosis', diag_kind='kde')
plt.savefig('plots/clinical/pair_plot.png')
plt.close()

# 3. Categorical Variable Analysis
categorical_vars = ['Gender', 'Country', 'Ethnicity', 'Family_History', 
                   'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 
                   'Obesity', 'Diabetes']

# Create stacked bar charts for each categorical variable
for var in categorical_vars:
    plt.figure(figsize=(12, 6))
    
    # Calculate percentages
    ct = pd.crosstab(df[var], df['Diagnosis'], normalize='index') * 100
    
    # Create stacked bar chart
    ct.plot(kind='bar', stacked=True)
    plt.title(f'Diagnosis Distribution by {var}')
    plt.xlabel(var)
    plt.ylabel('Percentage')
    plt.legend(title='Diagnosis')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'plots/categorical/{var}_analysis.png')
    plt.close()

# 4. Advanced Correlation Analysis
# Create correlation matrix with clustering
numerical_data = df[numerical_vars].copy()
correlation = numerical_data.corr()

# Generate clustered heatmap
plt.figure(figsize=(12, 10))
sns.clustermap(correlation, 
               annot=True, 
               cmap='coolwarm', 
               center=0,
               fmt='.2f',
               linewidths=0.5)
plt.savefig('plots/advanced/clustered_correlation.png')
plt.close()

# 5. Risk Factor Combinations Analysis
# Create risk score based on multiple factors
df['Risk_Count'] = (
    (df['Family_History'] == 'Yes').astype(int) +
    (df['Radiation_Exposure'] == 'Yes').astype(int) +
    (df['Iodine_Deficiency'] == 'Yes').astype(int) +
    (df['Smoking'] == 'Yes').astype(int) +
    (df['Obesity'] == 'Yes').astype(int) +
    (df['Diabetes'] == 'Yes').astype(int)
)

plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Risk_Count', y='Diagnosis')
plt.title('Risk Factor Count vs Diagnosis')
plt.savefig('plots/advanced/risk_count_analysis.png')
plt.close()

# 6. Age Group Analysis with Clinical Measurements
age_bins = [0, 30, 45, 60, 75, 100]
age_labels = ['<30', '30-45', '45-60', '60-75', '>75']
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

# Create subplot for each clinical measurement by age group
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
axes = axes.ravel()

for idx, var in enumerate(clinical_vars):
    sns.boxplot(data=df, x='Age_Group', y=var, hue='Diagnosis', ax=axes[idx])
    axes[idx].set_title(f'{var} by Age Group and Diagnosis')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/advanced/age_clinical_analysis.png')
plt.close()

# 7. Geographic Analysis
# Create choropleth map of cancer rates by country
country_rates = df.groupby('Country')['Diagnosis'].apply(
    lambda x: (x == 'Malignant').mean() * 100
).reset_index()
country_rates.columns = ['Country', 'Malignancy_Rate']

plt.figure(figsize=(15, 8))
sns.barplot(data=country_rates.sort_values('Malignancy_Rate', ascending=False),
            x='Country', y='Malignancy_Rate')
plt.title('Malignancy Rates by Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/advanced/geographic_analysis.png')
plt.close()

# 8. Multivariate Pattern Analysis
# Standardize numerical variables
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numerical_vars])
scaled_df = pd.DataFrame(scaled_data, columns=numerical_vars)

# Create radar chart for average values by diagnosis
fig = go.Figure()
for diagnosis in ['Benign', 'Malignant']:
    values = df[df['Diagnosis'] == diagnosis][numerical_vars].mean()
    values = pd.concat([values, values.iloc[0:1]])  # Complete the circle
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=numerical_vars + [numerical_vars[0]],
        name=diagnosis
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[-2, 2])),
    showlegend=True,
    title='Average Clinical Measurements by Diagnosis'
)
fig.write_html('plots/advanced/radar_chart.html')

# Print summary statistics
print("\nEnhanced EDA Summary:")
print("-" * 50)

# Summary of numerical variables by diagnosis
print("\nNumerical Variables Summary by Diagnosis:")
for var in numerical_vars:
    print(f"\n{var} Statistics:")
    print(df.groupby('Diagnosis')[var].describe())

# Chi-square tests for categorical variables
print("\nCategorical Variables Association with Diagnosis:")
for var in categorical_vars:
    contingency = pd.crosstab(df[var], df['Diagnosis'])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    print(f"\n{var}:")
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"p-value: {p_value:.2e}")

# Risk factor combinations
print("\nRisk Factor Combinations Analysis:")
risk_combinations = df.groupby('Risk_Count')['Diagnosis'].value_counts(normalize=True).unstack()
print("\nMalignancy Rate by Number of Risk Factors:")
print(risk_combinations['Malignant'].multiply(100).round(2))

print("\nAll visualizations have been saved in the plots directory with respective subdirectories.")

# Basic information about the dataset
print("\nDataset Info:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Distribution of categorical variables
print("\nDistribution of Categorical Variables:")
categorical_cols = ['Gender', 'Country', 'Ethnicity', 'Family_History', 
                   'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 
                   'Obesity', 'Diabetes', 'Thyroid_Cancer_Risk', 'Diagnosis']

for col in categorical_cols:
    print(f"\n{col} Distribution:")
    print(df[col].value_counts(normalize=True).round(3) * 100)

# 1. Age Distribution and Cancer Risk
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='Age', hue='Diagnosis', fill=True)
plt.title('Age Distribution by Diagnosis')
plt.savefig('plots/age_distribution.png')
plt.close()

# 2. Risk Factors Analysis
risk_factors = ['Family_History', 'Radiation_Exposure', 'Iodine_Deficiency', 
                'Smoking', 'Obesity', 'Diabetes']

plt.figure(figsize=(15, 8))
for i, factor in enumerate(risk_factors, 1):
    plt.subplot(2, 3, i)
    risk_ratio = df[df[factor] == 'Yes']['Diagnosis'].value_counts(normalize=True)
    no_risk_ratio = df[df[factor] == 'No']['Diagnosis'].value_counts(normalize=True)
    
    data = pd.DataFrame({
        'With Risk Factor': risk_ratio,
        'Without Risk Factor': no_risk_ratio
    }).T
    
    sns.heatmap(data, annot=True, fmt='.2%', cmap='RdYlGn_r')
    plt.title(f'{factor} Impact')

plt.tight_layout()
plt.savefig('plots/risk_factors_analysis.png')
plt.close()

# 3. Clinical Measurements Distribution
clinical_measures = ['TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']

plt.figure(figsize=(15, 8))
for i, measure in enumerate(clinical_measures, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, x='Diagnosis', y=measure)
    plt.title(f'{measure} by Diagnosis')

plt.tight_layout()
plt.savefig('plots/clinical_measurements.png')
plt.close()

# 4. Geographic and Demographic Analysis
plt.figure(figsize=(15, 6))

# Country-wise cancer rates
country_cancer_rates = df.groupby('Country')['Diagnosis'].apply(
    lambda x: (x == 'Malignant').mean()
).sort_values(ascending=False)

plt.subplot(1, 2, 1)
country_cancer_rates.plot(kind='bar')
plt.title('Cancer Rates by Country')
plt.xticks(rotation=45)

# Ethnicity-wise cancer rates
ethnicity_cancer_rates = df.groupby('Ethnicity')['Diagnosis'].apply(
    lambda x: (x == 'Malignant').mean()
).sort_values(ascending=False)

plt.subplot(1, 2, 2)
ethnicity_cancer_rates.plot(kind='bar')
plt.title('Cancer Rates by Ethnicity')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('plots/geographic_demographic_analysis.png')
plt.close()

# 5. Correlation Matrix of Numerical Variables
numerical_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Variables')
plt.savefig('plots/correlation_matrix.png')
plt.close()

# Advanced Statistical Analysis
print("\nAdvanced Statistical Analysis:")
print("-" * 50)

# Chi-square tests for categorical variables
categorical_vars = ['Gender', 'Family_History', 'Radiation_Exposure', 
                   'Iodine_Deficiency', 'Smoking', 'Obesity', 'Diabetes']

print("\nChi-square Test Results:")
for var in categorical_vars:
    contingency = pd.crosstab(df[var], df['Diagnosis'])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    print(f"\n{var}:")
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"p-value: {p_value:.2e}")

# Age Group Analysis
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 75, 100], 
                       labels=['<30', '30-45', '45-60', '60-75', '>75'])

age_risk = df.groupby('AgeGroup')['Diagnosis'].value_counts(normalize=True).unstack()
plt.figure(figsize=(10, 6))
age_risk['Malignant'].plot(kind='bar')
plt.title('Malignancy Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Malignancy Rate')
plt.tight_layout()
plt.savefig('plots/age_group_analysis.png')
plt.close()

# Clinical Measurements Analysis
clinical_vars = ['TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']

# T-tests for clinical measurements
print("\nT-test Results for Clinical Measurements:")
for var in clinical_vars:
    benign_vals = df[df['Diagnosis'] == 'Benign'][var]
    malignant_vals = df[df['Diagnosis'] == 'Malignant'][var]
    t_stat, p_value = stats.ttest_ind(benign_vals, malignant_vals)
    print(f"\n{var}:")
    print(f"T-statistic: {t_stat:.2f}")
    print(f"p-value: {p_value:.2e}")

# Machine Learning Feature Importance
# Prepare data for machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
le = LabelEncoder()
X = df[categorical_vars + clinical_vars].copy()
for col in categorical_vars:
    X[col] = le.fit_transform(X[col])
y = le.fit_transform(df['Diagnosis'])

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Feature importance plot
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title('Feature Importance in Predicting Thyroid Cancer')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()

# Risk Score Analysis
# Create a simple risk score based on significant features
df['RiskScore'] = (
    (df['Family_History'] == 'Yes').astype(int) * 2 +
    (df['Radiation_Exposure'] == 'Yes').astype(int) * 1.5 +
    (df['Iodine_Deficiency'] == 'Yes').astype(int) * 1.5 +
    (df['Nodule_Size'] > df['Nodule_Size'].median()).astype(int)
)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Diagnosis', y='RiskScore')
plt.title('Risk Score Distribution by Diagnosis')
plt.tight_layout()
plt.savefig('plots/risk_score_analysis.png')
plt.close()

# Print additional insights
print("\nMachine Learning Model Performance:")
print("-" * 50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# Risk Score Analysis
print("\nRisk Score Analysis:")
print("-" * 50)
print("\nRisk Score Statistics by Diagnosis:")
print(df.groupby('Diagnosis')['RiskScore'].describe())

# Calculate optimal risk score threshold
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y, df['RiskScore'])
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal Risk Score Threshold: {optimal_threshold:.2f}")
print(f"ROC AUC Score: {auc(fpr, tpr):.3f}")

# Save ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Risk Score')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('plots/roc_curve.png')
plt.close()

# Print some key statistics
print("\nKey Findings:")
print("-" * 50)

# Age statistics by diagnosis
print("\nAge Statistics by Diagnosis:")
print(df.groupby('Diagnosis')['Age'].describe())

# Risk factor impact
print("\nRisk Factor Impact on Malignancy Rate:")
for factor in risk_factors:
    malignancy_rate_with_risk = df[df[factor] == 'Yes']['Diagnosis'].value_counts(normalize=True)['Malignant']
    malignancy_rate_without_risk = df[df[factor] == 'No']['Diagnosis'].value_counts(normalize=True)['Malignant']
    print(f"\n{factor}:")
    print(f"With Risk Factor: {malignancy_rate_with_risk:.2%}")
    print(f"Without Risk Factor: {malignancy_rate_without_risk:.2%}")
    print(f"Relative Risk: {malignancy_rate_with_risk/malignancy_rate_without_risk:.2f}x")

print("\nPlots have been saved in the 'plots' directory.")
print("\nAll analyses complete. New plots have been saved in the 'plots' directory.")
