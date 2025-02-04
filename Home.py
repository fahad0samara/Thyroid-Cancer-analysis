import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import create_features

# Set page config
st.set_page_config(
    page_title="Thyroid Cancer Risk Predictor",
    page_icon="🏥",
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
    .stProgress .st-bo {
        background-color: #FF4B4B;
    }
    .risk-high {
        color: #FF4B4B;
        font-weight: bold;
        font-size: 20px;
    }
    .risk-medium {
        color: #FFA500;
        font-weight: bold;
        font-size: 20px;
    }
    .risk-low {
        color: #00CC96;
        font-weight: bold;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the models and preprocessors
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_thyroid_model.joblib')
        scaler = joblib.load('scaler.joblib')
        le = joblib.load('label_encoder.joblib')
        selected_features = joblib.load('selected_features.joblib')
        feature_names = joblib.load('feature_names.joblib')
        return model, scaler, le, selected_features, feature_names
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

model, scaler, le, selected_features, feature_names = load_models()

# Title and description
st.title("🏥 Thyroid Cancer Risk Assessment")
st.markdown("""
    This application uses machine learning to assess thyroid cancer risk based on various clinical and demographic factors.
    Please fill in the patient information below to get a risk assessment.
""")

# Create two columns for input
col1, col2 = st.columns(2)

# Define categorical options
GENDER_OPTIONS = ["Female", "Male"]
COUNTRY_OPTIONS = ["USA", "UK", "Canada", "Australia", "Other"]
ETHNICITY_OPTIONS = ["Caucasian", "African", "Asian", "Hispanic", "Other"]
YES_NO_OPTIONS = ["No", "Yes"]

with col1:
    st.subheader("📋 Patient Information")
    
    # Demographics
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    gender = st.selectbox("Gender", GENDER_OPTIONS)
    country = st.selectbox("Country", COUNTRY_OPTIONS)
    ethnicity = st.selectbox("Ethnicity", ETHNICITY_OPTIONS)
    
    # Risk Factors
    st.subheader("🔍 Risk Factors")
    family_history = st.selectbox("Family History of Thyroid Cancer", YES_NO_OPTIONS)
    radiation_exposure = st.selectbox("History of Radiation Exposure", YES_NO_OPTIONS)
    iodine_deficiency = st.selectbox("Iodine Deficiency", YES_NO_OPTIONS)

with col2:
    st.subheader("🩺 Clinical Measurements")
    
    # Clinical measurements
    tsh_level = st.number_input("TSH Level (mIU/L)", min_value=0.0, max_value=50.0, value=2.5)
    t3_level = st.number_input("T3 Level (ng/dL)", min_value=0.0, max_value=10.0, value=1.8)
    t4_level = st.number_input("T4 Level (μg/dL)", min_value=0.0, max_value=25.0, value=8.0)
    nodule_size = st.number_input("Nodule Size (cm)", min_value=0.0, max_value=10.0, value=1.5)
    
    # Additional risk factors
    smoking = st.selectbox("Smoking Status", YES_NO_OPTIONS)
    obesity = st.selectbox("Obesity", YES_NO_OPTIONS)
    diabetes = st.selectbox("Diabetes", YES_NO_OPTIONS)

# Create prediction function
def predict_thyroid_cancer(patient_data):
    try:
        # Create DataFrame with initial features
        df = pd.DataFrame([patient_data])
        
        # Apply feature engineering using the same function as training
        df = create_features(df)
        
        # Ensure we have all required features in the correct order
        df_final = pd.DataFrame(index=df.index)
        
        # Fill in missing features with 0 if they don't exist
        for feature in feature_names:
            if feature not in df.columns:
                df_final[feature] = 0
            else:
                df_final[feature] = df[feature]
        
        # Encode categorical variables
        categorical_vars = ['Gender', 'Country', 'Ethnicity', 'Family_History',
                          'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking',
                          'Obesity', 'Diabetes', 'Age_Group']
        
        # Create a mapping dictionary for each categorical variable
        category_maps = {
            'Gender': {val: idx for idx, val in enumerate(GENDER_OPTIONS)},
            'Country': {val: idx for idx, val in enumerate(COUNTRY_OPTIONS)},
            'Ethnicity': {val: idx for idx, val in enumerate(ETHNICITY_OPTIONS)},
            'Family_History': {val: idx for idx, val in enumerate(YES_NO_OPTIONS)},
            'Radiation_Exposure': {val: idx for idx, val in enumerate(YES_NO_OPTIONS)},
            'Iodine_Deficiency': {val: idx for idx, val in enumerate(YES_NO_OPTIONS)},
            'Smoking': {val: idx for idx, val in enumerate(YES_NO_OPTIONS)},
            'Obesity': {val: idx for idx, val in enumerate(YES_NO_OPTIONS)},
            'Diabetes': {val: idx for idx, val in enumerate(YES_NO_OPTIONS)},
            'Age_Group': {val: idx for idx, val in enumerate(['<30', '30-45', '45-60', '60-75', '>75'])}
        }
        
        # Encode categorical variables
        for col in categorical_vars:
            df_final[col] = df_final[col].map(category_maps[col])
        
        # Scale numerical features
        numerical_features = [col for col in feature_names if col not in categorical_vars]
        df_final[numerical_features] = scaler.transform(df_final[numerical_features])
        
        # Select only the features used in training
        df_final = df_final[selected_features]
        
        # Make prediction
        prob = model.predict_proba(df_final)[0, 1]
        prediction = 'Malignant' if prob > 0.5 else 'Benign'
        risk_level = 'High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low'
        
        return prediction, prob, risk_level
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

# Prediction button
if st.button("Generate Risk Assessment"):
    # Create progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
    
    # Collect patient data
    patient_data = {
        'Age': age,
        'Gender': gender,
        'Country': country,
        'Ethnicity': ethnicity,
        'Family_History': family_history,
        'Radiation_Exposure': radiation_exposure,
        'Iodine_Deficiency': iodine_deficiency,
        'Smoking': smoking,
        'Obesity': obesity,
        'Diabetes': diabetes,
        'TSH_Level': tsh_level,
        'T3_Level': t3_level,
        'T4_Level': t4_level,
        'Nodule_Size': nodule_size
    }
    
    # Make prediction
    prediction, probability, risk_level = predict_thyroid_cancer(patient_data)
    
    if prediction and probability and risk_level:
        # Display results
        st.markdown("---")
        st.subheader("📊 Risk Assessment Results")
        
        # Create three columns for results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Prediction")
            st.markdown(f"<div class='risk-{'high' if prediction == 'Malignant' else 'low'}'>{prediction}</div>",
                       unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Risk Level")
            st.markdown(f"<div class='risk-{risk_level.lower()}'>{risk_level}</div>",
                       unsafe_allow_html=True)
        
        with col3:
            st.markdown("### Probability")
            st.markdown(f"<div style='font-size: 20px; font-weight: bold;'>{probability:.1%}</div>",
                       unsafe_allow_html=True)
        
        # Create gauge chart for risk visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'green'},
                    {'range': [30, 70], 'color': 'orange'},
                    {'range': [70, 100], 'color': 'red'}
                ],
            }
        ))
        
        st.plotly_chart(fig)
        
        # Risk factor analysis
        st.markdown("---")
        st.subheader("🔍 Risk Factor Analysis")
        
        # Create risk factor chart
        risk_factors = {
            'Family History': 2 if family_history == 'Yes' else 0,
            'Radiation Exposure': 1.5 if radiation_exposure == 'Yes' else 0,
            'Iodine Deficiency': 1.5 if iodine_deficiency == 'Yes' else 0,
            'Smoking': 1 if smoking == 'Yes' else 0,
            'Obesity': 1 if obesity == 'Yes' else 0,
            'Diabetes': 1 if diabetes == 'Yes' else 0
        }
        
        fig = px.bar(
            x=list(risk_factors.keys()),
            y=list(risk_factors.values()),
            title="Risk Factor Contributions",
            labels={'x': 'Risk Factor', 'y': 'Risk Weight'}
        )
        fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                         marker_line_width=1.5, opacity=0.6)
        st.plotly_chart(fig)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        recommendations = []
        if family_history == "Yes":
            recommendations.append("• Regular screening due to family history")
        if radiation_exposure == "Yes":
            recommendations.append("• Close monitoring due to radiation exposure history")
        if smoking == "Yes":
            recommendations.append("• Consider smoking cessation program")
        if obesity == "Yes":
            recommendations.append("• Weight management consultation recommended")
        if iodine_deficiency == "Yes":
            recommendations.append("• Dietary iodine supplementation may be needed")
        if tsh_level > 4.0 or tsh_level < 0.4:
            recommendations.append("• Further thyroid function testing recommended")
        if nodule_size > 1.0:
            recommendations.append("• Regular ultrasound monitoring of nodule size")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(rec)
        else:
            st.markdown("• Continue routine thyroid health monitoring")

# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: grey;'>
        Developed for medical research purposes. Always consult with healthcare professionals for medical advice.
    </div>
""", unsafe_allow_html=True)
