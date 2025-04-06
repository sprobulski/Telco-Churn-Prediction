from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,classification_report,roc_curve, auc
from lime.lime_tabular import LimeTabularExplainer
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import numpy as np
import shap

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


def get_results(y_test,y_pred,y_train,y_pred_train):
    """
    Compute and display classification metrics, a classification report, and a confusion matrix for train and test predictions.
    
    Parameters:
    - y_test: True labels for the test set (pandas Series or numpy array)
    - y_pred: Predicted labels for the test set (numpy array)
    - y_train: True labels for the training set (pandas Series or numpy array)
    - y_pred_train: Predicted labels for the training set (numpy array)
    
    Functionality:
    - Prints accuracy, precision, recall, F1 score, and AUC for both test and train sets using sklearn.metrics functions.
    - Displays a detailed classification report for the test set, including per-class metrics.
    - Plots a confusion matrix heatmap for the test set using seaborn, showing true vs. predicted labels.
    """


    print("Metrics for test and train")
    print("Metrics for test")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1: {f1_score(y_test, y_pred):.4f}")

    print("\nMetrics for train")
    print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Precision: {precision_score(y_train, y_pred_train):.4f}")
    print(f"Recall: {recall_score(y_train, y_pred_train):.4f}")
    print(f"F1: {f1_score(y_train, y_pred_train):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc_curve(model, X_test, y_test, title="ROC Curve"):
    """
    Plot ROC curve for a single model.
    
    Parameters:
    - model: Trained model (e.g., LogisticRegression, XGBClassifier, Keras model)
    - X_test: Test features (pandas DataFrame or numpy array)
    - y_test: True test labels (pandas Series or numpy array)
    - title: Plot title (string, default="ROC Curve")


    Functionality:
    - Computes the ROC curve and AUC using sklearn.metrics functions.
    - Plots the ROC curve using matplotlib, showing the true positive rate vs. false positive rate.
    """
    # Get predicted probabilities
    if hasattr(model, 'predict_proba'):  # For sklearn models
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:  # For Keras/TF neural networks
        y_pred_proba = model.predict(X_test, verbose=0).ravel()
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Print AUC
    print(f"AUC: {roc_auc:.4f}")

def fit_and_get_logistic_coefficients(X_train, X_test, y_train, y_test):
    """
    Fit a logistic regression pipeline and return the trained model and sorted coefficients.
    
    Parameters:
    - X_train: Training features (pandas DataFrame)
    - X_test: Test features (pandas DataFrame)
    - y_train: Training labels (pandas Series or numpy array)
    - y_test: Test labels (pandas Series or numpy array)
    
    Returns:
    - logistic_pipeline: Trained Pipeline object containing scaler and logistic regression
    - coef_df: DataFrame with feature names and sorted coefficients
    """
    # Define and fit the pipeline
    logistic_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegressionCV(
            solver='liblinear', penalty='l1', random_state=42, max_iter=1000, cv=5, class_weight='balanced'))
    ])
    logistic_pipeline.fit(X_train, y_train)

    # Extract the logistic regression model
    model_logistic = logistic_pipeline.named_steps['logistic']

    # Get predictions
    y_pred = logistic_pipeline.predict(X_test)
    y_pred_train = logistic_pipeline.predict(X_train)

    # Get coefficients and feature names
    coefficients = model_logistic.coef_[0]
    feature_names = X_train.columns

    # Create and sort coefficient DataFrame
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)


    return model_logistic, coef_df,y_pred,y_pred_train

def fit_and_get_xgb_model(X_train, y_train,save_model=False):
    """
    Fit an XGBoost model and return the trained model and predictions.

    Parameters:
    - X_train: Training features (pandas DataFrame)
    - y_train: Training labels (pandas Series or numpy array)
    - save_model: Whether to save the model after training (boolean, default=False)



    Returns:
    - grid_search: Trained XGBoost model after hyperparameter tuning
    """

    model = xgb.XGBClassifier(
    objective = 'binary:logistic',
    eval_metric = 'logloss',
    random_state=42,
    max_delta_step=1,
    )
    param_grid = {
    'learning_rate': [0.01,0.05, 0.1],
    'max_depth': [2],
    'n_estimators': [100,150,200],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'min_child_weight': [5],
    'reg_alpha': [0.1,1,10],
    'reg_lambda': [0.1,1,10],
    'scale_pos_weight': [2,2.76]

    }
    # Grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    # Results
    print("Best Parameters:", grid_search.best_params_)
    print("Best CV F1 Score:", grid_search.best_score_)


    #save the model
    if save_model:
        grid_search.best_estimator_.save_model('../models/model_xgb.json')
        print("Model saved as model_xgb.json")


    return grid_search


def fit_and_get_neural_network_model(X_train, y_train, epochs=100,batch_size=32,patience=10,save_model=False):
    """
    Fit a neural network model and return the trained model and training history.


    Parameters:
    - X_train: Training features (numpy array or pandas DataFrame)
    - y_train: Training labels (numpy array or pandas Series)
    - epochs: Number of epochs for training (integer, default=100)
    - batch_size: Batch size for training (integer, default=32)
    - patience: Number of epochs with no improvement after which training will be stopped (integer, default=10)
    - save_model: Whether to save the model after training (boolean, default=False)
    

    Returns:
    - model: Trained Keras Sequential model
    - history: Training history object containing loss and metric values during training
    """
    # Define the model
    model = Sequential()
    model.add(Dense(128,activation='relu',input_shape = (X_train.shape[1],)))
    model.add(Dropout(0.4))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(1,activation='relu'))

    weight = (len(y_train[y_train==0]))/len(y_train[y_train==1])
    class_weights = {0:1,1:weight}
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['f1_score'])

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Train
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weights,
                        validation_split=0.2, callbacks=[lr_scheduler, early_stopping], verbose=1)
    if save_model:
        model.save(f"../models/NeuralNetwork_Churn.h5")
        print(f"model saved as 'NeuralNetwork_Churn.h5'")
    return model, history


def plot_lime_explanation(model, X_train, X_test, observation_idx, num_features=19, title="LIME Explanation"):
    """
    Plot LIME explanation for a single observation from a trained model.
    
    Parameters:
    - model: Trained model (e.g., logistic_pipeline with predict_proba support)
    - X_train: Training features (pandas DataFrame) for explainer initialization
    - X_test: Test features (pandas DataFrame or numpy array) for prediction
    - observation_idx: Index of the observation in X_test to explain (integer)
    - num_features: Number of top features to display (integer, default=10)
    - title: Plot title (string, default="LIME Explanation")
    """
    # Convert X_train to numpy array if it’s a DataFrame (LIME requires numpy)
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test

    # Create LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=X_train.columns.tolist(),
        class_names=['No Churn', 'Churn'],
        mode='classification'
    )

    exp = explainer.explain_instance(
        data_row=X_test_np[observation_idx],
        predict_fn=model.predict_proba,
        num_features=num_features
    )

    fig = exp.as_pyplot_figure()
    plt.title(f"{title} - Observation {observation_idx}")
    plt.tight_layout()
    plt.xlabel('Feature contribution to the prediction')
    plt.show()

    pred_proba = model.predict_proba(X_test_np)[observation_idx]
    print(f"Predicted Probabilities: No Churn = {pred_proba[0]:.4f}, Churn = {pred_proba[1]:.4f}")


def plot_xgb_feature_importance(model, X_train, title="Feature Importance - XGBoost"):
    """
    Plot feature importance for an XGBoost model using X_train column names.
    
    Parameters:
    - model: Trained XGBoost model (e.g., model_xgb.best_estimator_)
    - X_train: Training features (pandas DataFrame) with column names
    - title: Plot title (string, default="Feature Importance - XGBoost")
    """
    # Get importance and feature names
    importance = model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(title)
    plt.gca().invert_yaxis()  # Top features at the top
    plt.tight_layout()
    plt.show()

    # Print for reference
    print("Feature Importance Scores:")
    print(feature_importance_df)

def plot_shap(model, model_type, scaler, X_test, feature_name, observation_idx, output_idx=0):
    """
    Plot SHAP values (summary bar, summary dot, dependence, waterfall) for a trained model and a specific observation.

    Parameters:
    - model: Trained model (e.g., logistic_pipeline, xgb_model, or neural network)
    - model_type: Type of model ('logistic', 'xgboost', 'neural network')
    - scaler: Scaler used for preprocessing (e.g., StandardScaler)
    - X_test: Test features (pandas DataFrame or numpy array)
    - feature_name: Name of the feature for dependence plot (string)
    - observation_idx: Index of the observation in X_test to explain (integer)
    - output_idx: Index of the output to analyze for neural networks (default 0, ignored for other models)
    """
    # Convert X_test to DataFrame if it’s a NumPy array
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
    
    # Rescale X_test for interpretability
    X_test_rescaled = scaler.inverse_transform(X_test)
    X_test_rescaled_df = pd.DataFrame(X_test_rescaled, columns=X_test.columns)

    # Compute SHAP values based on model type
    if model_type == 'logistic':
        # For logistic regression, use the model directly
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(X_test)
        base_value = explainer.expected_value
        shap_values_subset = shap_values  # Use full dataset for logistic
        X_test_subset = X_test
    elif model_type == 'xgboost':
        # For XGBoost, use TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        base_value = explainer.expected_value
        shap_values_subset = shap_values  # Use full dataset for XGBoost
        X_test_subset = X_test
    elif model_type == 'neural network':
        # For neural networks, use KernelExplainer with a subset for speed
        X_test_subset = X_test.iloc[:50, :]  # Small subset for summary/dependence
        background = X_test_subset.copy()

        # Define prediction function to handle multi-output
        def predict_fn(X):
            pred = model.predict(X)
            return pred[:, output_idx] if pred.ndim > 1 else pred  # Force single output

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values_subset = explainer.shap_values(X_test_subset, nsamples=100)  # Fast computation
        shap_values_single = explainer.shap_values(X_test.iloc[observation_idx:observation_idx+1], nsamples=100)
        base_value = explainer.expected_value
        shap_values = shap_values_single  # For waterfall, use single observation
    else:
        raise ValueError("Unsupported model type. Use 'logistic', 'xgboost', or 'neural network'.")


    # 1. Plot SHAP summary plot (bar)
    shap.summary_plot(shap_values_subset, X_test_subset, feature_names=X_test.columns, plot_type="bar", show=False)
    plt.title(f"SHAP Summary Plot{' - Output ' + str(output_idx) if model_type == 'neural network' else ''}")
    plt.show()

    # 2. Plot SHAP feature impact (dot)
    shap.summary_plot(shap_values_subset, X_test_subset, feature_names=X_test.columns, show=False)
    plt.title(f"SHAP Feature Impact - Dot{' - Output ' + str(output_idx) if model_type == 'neural network' else ''}")
    plt.tight_layout()
    plt.show()

    # 3. Plot SHAP dependence plot for the specified feature
    shap.dependence_plot(
        ind=feature_name,
        shap_values=shap_values_subset,
        features=X_test_subset,
        feature_names=X_test.columns,
        show=False
    )
    plt.title(f"SHAP Dependence Plot - {feature_name}{' - Output ' + str(output_idx) if model_type == 'neural network' else ''}")
    plt.tight_layout()
    plt.show()

    # 4. Plot SHAP waterfall plot for the single observation
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[observation_idx] if model_type != 'neural network' else shap_values_single[0],
            base_values=base_value,
            data=X_test_rescaled_df.iloc[observation_idx],
            feature_names=X_test.columns.tolist()
        ),
        show=False
    )
    plt.title(f"SHAP Waterfall Plot - Observation {observation_idx}{' - Output ' + str(output_idx) if model_type == 'neural network' else ''}")
    plt.tight_layout()
    plt.show()