import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Thyroid Cancer Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('thyroid_cancer_risk_data.csv')
    return df

df = load_data()

# Title
st.title("📊 Thyroid Cancer Data Analysis")
st.markdown("""
    This page provides comprehensive analysis of thyroid cancer data including distributions,
    correlations, and risk factor analysis.
""")

# Sidebar for analysis options
st.sidebar.header("Analysis Options")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Overview", "Clinical Measurements", "Risk Factors", "Demographics", "Advanced Analysis"]
)

if analysis_type == "Overview":
    st.header("📈 Data Overview")
    
    # Dataset statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Malignant Cases", len(df[df['Diagnosis'] == 'Malignant']))
    with col3:
        st.metric("Benign Cases", len(df[df['Diagnosis'] == 'Benign']))
    
    # Distribution of diagnosis
    fig = px.pie(df, names='Diagnosis', title='Distribution of Diagnosis',
                color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig)
    
    # Age distribution by diagnosis
    fig = px.histogram(df, x='Age', color='Diagnosis', nbins=30,
                    title='Age Distribution by Diagnosis',
                    marginal='box')
    st.plotly_chart(fig)

elif analysis_type == "Clinical Measurements":
    st.header("🔬 Clinical Measurements Analysis")
    
    # Select measurement
    measurement = st.selectbox(
        "Select Measurement",
        ["TSH_Level", "T3_Level", "T4_Level", "Nodule_Size"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot
        fig = px.box(df, x='Diagnosis', y=measurement,
                    title=f'{measurement} by Diagnosis',
                    color='Diagnosis')
        st.plotly_chart(fig)
    
    with col2:
        # Violin plot
        fig = px.violin(df, x='Diagnosis', y=measurement,
                    title=f'{measurement} Distribution',
                    color='Diagnosis', box=True)
        st.plotly_chart(fig)
    
    # Correlation heatmap
    clinical_vars = ['TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']
    corr = df[clinical_vars].corr()
    fig = px.imshow(corr, title='Correlation Matrix of Clinical Measurements',
                    labels=dict(color="Correlation"))
    st.plotly_chart(fig)

elif analysis_type == "Risk Factors":
    st.header("⚠️ Risk Factor Analysis")
    
    risk_factors = ['Family_History', 'Radiation_Exposure', 'Iodine_Deficiency',
                    'Smoking', 'Obesity', 'Diabetes']
    
    # Risk factor distribution
    fig = go.Figure()
    for factor in risk_factors:
        risk_dist = df.groupby([factor, 'Diagnosis']).size().unstack(fill_value=0)
        risk_dist_pct = risk_dist.div(risk_dist.sum(axis=1), axis=0) * 100
        
        fig.add_trace(go.Bar(
            name=factor,
            x=[factor],
            y=[risk_dist_pct.loc['Yes', 'Malignant']],
            text=[f"{risk_dist_pct.loc['Yes', 'Malignant']:.1f}%"],
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Malignancy Rate by Risk Factor',
        yaxis_title='Percentage of Malignant Cases',
        showlegend=False
    )
    st.plotly_chart(fig)
    
    # Risk factor combinations
    st.subheader("Risk Factor Combinations")
    df['Risk_Count'] = df[risk_factors].apply(lambda x: (x == 'Yes').sum(), axis=1)
    
    fig = px.histogram(df, x='Risk_Count', color='Diagnosis',
                    title='Number of Risk Factors vs Diagnosis',
                    labels={'Risk_Count': 'Number of Risk Factors'},
                    barmode='group')
    st.plotly_chart(fig)

elif analysis_type == "Demographics":
    st.header("👥 Demographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution
        fig = px.pie(df, names='Gender', title='Gender Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig)
        
        # Age groups
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 75, 100],
                                labels=['<30', '30-45', '45-60', '60-75', '>75'])
        fig = px.histogram(df, x='Age_Group', color='Diagnosis',
                        title='Age Groups by Diagnosis',
                        barmode='group')
        st.plotly_chart(fig)
    
    with col2:
        # Ethnicity distribution
        fig = px.pie(df, names='Ethnicity', title='Ethnicity Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig)
        
        # Country distribution
        fig = px.pie(df, names='Country', title='Country Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig)

else:  # Advanced Analysis
    st.header("🔍 Advanced Analysis")
    
    # Create composite score
    df['Thyroid_Function_Score'] = (
        (df['TSH_Level'] - df['TSH_Level'].mean()) / df['TSH_Level'].std() +
        (df['T3_Level'] - df['T3_Level'].mean()) / df['T3_Level'].std() +
        (df['T4_Level'] - df['T4_Level'].mean()) / df['T4_Level'].std()
    )
    
    # 3D scatter plot
    fig = px.scatter_3d(df, x='TSH_Level', y='T3_Level', z='T4_Level',
                        color='Diagnosis', size='Nodule_Size',
                        title='3D Visualization of Thyroid Function Tests')
    st.plotly_chart(fig)
    
    # Risk score analysis
    st.subheader("Risk Score Analysis")
    
    # Calculate risk score
    df['Risk_Score'] = (
        (df['Family_History'] == 'Yes').astype(int) * 2 +
        (df['Radiation_Exposure'] == 'Yes').astype(int) * 1.5 +
        (df['Iodine_Deficiency'] == 'Yes').astype(int) * 1.5 +
        (df['Smoking'] == 'Yes').astype(int) +
        (df['Obesity'] == 'Yes').astype(int) +
        (df['Diabetes'] == 'Yes').astype(int)
    )
    
    fig = px.box(df, x='Diagnosis', y='Risk_Score',
                color='Diagnosis', title='Risk Score Distribution by Diagnosis',
                points='all')
    st.plotly_chart(fig)
    
    # Thyroid function score vs risk score
    fig = px.scatter(df, x='Risk_Score', y='Thyroid_Function_Score',
                    color='Diagnosis', size='Nodule_Size',
                    title='Thyroid Function Score vs Risk Score',
                    trendline="ols")
    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: grey;'>
        Analysis based on the thyroid cancer risk dataset. For research purposes only.
    </div>
""", unsafe_allow_html=True)
