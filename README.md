# Thyroid Cancer Risk Prediction Model

An advanced machine learning application for predicting thyroid cancer risk using multiple models and comprehensive feature engineering.

## Features

- **Advanced ML Models**:
  - XGBoost (optimized with Optuna)
  - LightGBM
  - CatBoost
  - Voting Ensemble

- **Feature Engineering**:
  - Hormone balance scoring
  - Clinical severity assessment
  - Risk factor combinations
  - Polynomial features
  - Age group categorization

- **Interactive Web Interface**:
  - Real-time predictions
  - Risk visualization
  - Personalized recommendations
  - Medical insights

## Project Structure

```
├── Home.py                     # Main Streamlit application
├── pages/                      # Streamlit pages
│   └── 1_Data_Analysis.py     # Data analysis dashboard
├── feature_engineering.py      # Feature engineering functions
├── train_advanced_model.py     # Model training script
├── train_advanced_model.ipynb  # Jupyter notebook version
├── analyze_thyroid_data.py     # Data analysis utilities
├── thyroid_cancer_ml.py        # Core ML functions
├── plots/                      # Generated visualizations
│   ├── distributions/          # Feature distribution plots
│   ├── correlation_matrix.png  # Feature correlations
│   ├── feature_importance.png  # Model feature importance
│   └── risk_factors_analysis.png # Risk factor analysis
├── requirements.txt           # Project dependencies
└── README.md                 # Project documentation
```

## File Descriptions

### Core Components

- **Home.py**: Main Streamlit application that provides:
  - Patient information input
  - Risk assessment calculation
  - Interactive visualizations
  - Medical recommendations

- **feature_engineering.py**: Contains all feature engineering functions:
  - Hormone ratio calculations
  - Risk score computation
  - Age group categorization
  - Clinical severity assessment
  - Polynomial feature generation

- **train_advanced_model.py**: Model training pipeline:
  - Data preprocessing
  - Feature selection
  - Model optimization
  - Ensemble creation
  - Model evaluation

### Analysis Tools

- **analyze_thyroid_data.py**: Data analysis utilities:
  - Statistical analysis
  - Data visualization
  - Feature correlation analysis
  - Distribution analysis

- **thyroid_cancer_ml.py**: Machine learning utilities:
  - Model definitions
  - Training functions
  - Evaluation metrics
  - Prediction functions

### Streamlit Components

The application is built using Streamlit and consists of multiple pages:

1. **Main Page (Home.py)**:
   - Patient data input form
   - Real-time risk assessment
   - Interactive risk visualizations
   - Personalized recommendations

2. **Data Analysis (pages/1_Data_Analysis.py)**:
   - Population statistics
   - Risk factor distributions
   - Correlation analysis
   - Feature importance visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fahad0samara/Thyroid-Cancer-analysis.git
cd Thyroid-Cancer-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web App

Start the Streamlit application:
```bash
streamlit run Home.py
```

The application will open in your default web browser with these features:
- Input patient information
- View real-time risk assessment
- Explore risk factor analysis
- Get personalized recommendations

### Training New Models

Train a new model using either:

```bash
# Using Python script
python train_advanced_model.py

# Using Jupyter Notebook
jupyter notebook train_advanced_model.ipynb
```

## Model Performance

Current model metrics:
- ROC AUC Score: 0.70
- Precision (High Risk): 0.71
- Recall (High Risk): 0.45
- Overall Accuracy: 0.83

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: Thyroid Cancer Risk Factors Dataset
- Medical research papers and guidelines used in feature engineering
- Open-source ML libraries: XGBoost, LightGBM, CatBoost
- Streamlit for the web interface
