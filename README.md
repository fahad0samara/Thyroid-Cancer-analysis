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

- **User Interface**:
  - Interactive Streamlit web application
  - Real-time risk assessment
  - Visual risk factor analysis
  - Personalized recommendations

## Project Structure

```
├── Home.py                    # Streamlit web application
├── train_advanced_model.py    # Model training script
├── train_advanced_model.ipynb # Jupyter notebook version
├── feature_engineering.py     # Feature engineering functions
├── requirements.txt           # Project dependencies
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/thyroid-cancer-prediction.git
cd thyroid-cancer-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script:
```bash
python train_advanced_model.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook train_advanced_model.ipynb
```

### Running the Web App

Start the Streamlit application:
```bash
streamlit run Home.py
```

## Model Details

The system uses an ensemble of three powerful gradient boosting models:
- XGBoost: Optimized using Optuna for hyperparameter tuning
- LightGBM: Known for handling categorical features effectively
- CatBoost: Robust against overfitting
- Voting Ensemble: Combines predictions from all models

## Feature Engineering

The model uses sophisticated feature engineering techniques:
- Hormone ratios (TSH/T3, TSH/T4, T3/T4)
- Risk scoring based on medical factors
- Age group categorization
- Clinical severity assessment
- Polynomial feature interactions

## Performance

The model achieves:
- ROC AUC Score: ~0.70
- High precision in identifying high-risk cases
- Balanced performance across different risk levels

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [Add source information]
- Medical guidelines and research papers used in feature engineering
- Open-source ML libraries: XGBoost, LightGBM, CatBoost
