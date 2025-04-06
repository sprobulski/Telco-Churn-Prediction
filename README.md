# Telco-Churn-Prediction
## Project Overview

This project predicts customer churn for a telecommunications company using the `Telco-Customer-Churn-Encoded.csv` dataset. The goal is to identify at-risk customers and provide actionable insights through three machine learning models: **XGBoost** (tree-based), **Neural Network** (deep learning), and **Logistic Regression** (linear). The work includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and explainability analysis.

This project demonstrates skills in:
- Classification and predictive modeling
- Data preprocessing and imbalance handling
- Machine learning with XGBoost, Neural Networks, and Logistic Regression
- Python programming (pandas, scikit-learn, TensorFlow, xgboost)
- Model interpretability (SHAP, LIME, feature importance)

---
## Dataset

The `Telco-Customer-Churn-Encoded.csv` dataset contains 7,032 customer records with 20 encoded features from a telecommunications company, capturing demographics, services, billing, and contract details. The target variable, `Churn`, is binary (0 = no churn, 1 = churn), with a dataset-wide churn rate of 26.58%.

### Data Structure
- **Continuous Features**: `tenure` (1-72 months, mean 32.42), `MonthlyCharges` (mean $64.80, median $70.35), `TotalCharges`.
- **Binary Features**: `gender` (~50% each), `SeniorCitizen` (16.24%), `Partner` (48.25%), `Dependents` (29.85%), `PhoneService` (90.33%), `PaperlessBilling` (59.27%), `Churn`.
- **Categorical Features**: `Contract` (55.11% month-to-month), `InternetService` (44.03% fiber optic), `PaymentMethod` (33.63% electronic check), and service add-ons (e.g., `OnlineSecurity`).
- **Preparation**: No missing values; categorical variables encoded. Models use varied preprocessing: XGBoost with SMOTE (training churn balanced to 40%) and `MinMaxScaler`, Neural Network with `StandardScaler` (drops `PhoneService`, `gender`), Logistic Regression with `MinMaxScaler`.

### EDA Findings
- **Churn Distribution**: Imbalanced (73.4% no churn, 26.6% churn in training split), addressed via SMOTE for XGBoost.
- **Customer Profile**: Half have tenure <29 months; higher monthly charges skew toward $70+. Seniors, singles, and paperless billing users are prevalent among churners.
- **Statistical Insights**:
  - **Continuous**: Churners have shorter `tenure`, higher `MonthlyCharges`, and lower `TotalCharges` (Mann-Whitney U, p<0.0001), suggesting early exits due to cost or dissatisfaction.
  - **Categorical**: Month-to-month contracts, fiber optic internet, and electronic checks drive churn (Chi-Squared, p<0.0001). Lack of add-ons (e.g., `OnlineSecurity`) and streaming services correlate with leaving; `gender` and `PhoneService` are neutral (p>0.05).
- **Correlations**: `tenure` negatively tied to churn (longer stays reduce risk); `MonthlyCharges` positively linked (higher costs increase it).
- **Key Drivers**: Short tenure, high costs, and lack of long-term contracts or stabilizing services fuel churn, especially for fiber optic users.
  
---

## Methodology

### 1. Data Preprocessing
- Split data: 70% train, 30% test (`random_state=42`).
- Applied scaling: `MinMaxScaler` (XGBoost, Logistic Regression), `StandardScaler` (Neural Network).
- Balanced training data for XGBoost using SMOTE (`sampling_strategy=0.6666`).

### 2. Exploratory Data Analysis (EDA)
- **Churn Distribution**: Confirmed imbalance (73.4% no churn, 26.6% churn), addressed with SMOTE for XGBoost.
- **Feature Relevance**: `tenure`, `Contract`, and `TotalCharges` identified as key predictors via model outputs.

### 3. Modeling
- **XGBoost**:
  - Tree-based model trained with grid search (648 candidates, 5-fold CV) to optimize parameters (e.g., `max_depth=2`, `n_estimators=200`, `learning_rate=0.1`, `colsample_bytree=0.8`).
  - Used SMOTE-balanced data; saved as `model_xgb.json`.
  - Predictions evaluated with accuracy, precision, recall, F1, and AUC-ROC (plot generated).
- **Neural Network**:
  - Built with TensorFlow/Keras; architecture unspecified but trained for 100 epochs, batch size 32, with early stopping (patience=10).
  - Dropped `PhoneService` and `gender`; optimized classification threshold using F1 score from precision-recall curve.
  - Evaluated with accuracy, precision, recall, F1, and AUC-ROC (plot generated); saved as `NeuralNetwork_Churn.h5`.
- **Logistic Regression**:
  - Linear model trained with scikit-learn, producing interpretable coefficients (e.g., `tenure`: -1.36, `TotalCharges`: 0.67).
  - Predictions assessed with accuracy, precision, recall, F1, and AUC-ROC (plot generated).
- **Evaluation**: All models assessed on test data (30%) with metrics and ROC curves plotted.


### 4. Explainability
- **SHAP** (XGBoost, Neural Network):
  - Generated summary plots, dot plots, dependence plots (plottable for any variable, e.g., `tenure`), and waterfall plots (single observation).
  - XGBoost: Most influential features are `Contract`, `tenure`, and `MonthlyCharges`.
  - Neural Network: `tenure` dominates, followed by `Contract` and `MonthlyCharges`.
- **LIME** (XGBoost, Logistic Regression):
  - Explained single predictions with feature contributions and probabilities.
  - XGBoost example: No Churn 0.9667, Churn 0.0333 (observation 0).
  - Logistic Regression example: No Churn 0.8417, Churn 0.1583 (observation 1).
- **Feature Importance** (XGBoost):
  - Plotted to highlight `Contract`, `OnlineSecurity`, and `TechSupport` as top contributors.

---

## Key Findings
- **Feature Impact**: Longer `tenure` and `Contract` reduce churn likelihood; higher `MonthlyCharges` and `TotalCharges` increase it. Lack of add-ons (e.g., `OnlineSecurity`) also drives churn.
- **Model Performance**:
  - **XGBoost**: Accuracy 0.7445, Precision 0.5123, Recall 0.8164, F1 0.6296, AUC (plotted). Excels at identifying churners due to SMOTE and robust feature handling.
  - **Neural Network**: Accuracy 0.7749, Precision 0.5569, Recall 0.7504, F1 0.6393, AUC (plotted). Best overall balance, leveraging tenure’s predictive power.
  - **Logistic Regression**: Accuracy 0.7427, Precision 0.5101, Recall 0.8128, F1 0.6268, AUC (plotted). High recall and interpretable coefficients.
- **Explainability Insights**: SHAP confirms `tenure` and `Contract` as critical across models; XGBoost feature importance adds `OnlineSecurity` and `TechSupport` as retention factors.

---
## Future Improvements
- **Hyperparameter Tuning**: Optimize Neural Network and XGBoost further with grid search.
- **Ensemble Methods**: Combine models for improved performance.

## Usage
To explore the project:
- Run preprocessing and EDA scripts
- Train models with respective scripts: XGBoost (`fit_and_get_xgb_model`), Neural Network (`fit_and_get_neural_network_model`), Logistic Regression (`fit_and_get_logistic_coefficients`).

---

## Dependencies
See `requirements.txt` for a full list. Key libraries include:
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`, `xgboost`: Machine learning
- `tensorflow`: Neural Network
- `imblearn`: SMOTE for imbalance
- `matplotlib`, `seaborn`: Visualization (implied)
- Custom `src.modeling_functions`: Utility functions
