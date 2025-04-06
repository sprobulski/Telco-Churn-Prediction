import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu,chi2_contingency

binary_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
categorical_features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
continous_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

def clean_data(data):
    data = data.drop(columns = 'customerID')
    mask = data['TotalCharges'] == ' '
    data = data[~mask]
    return data
    
def encode_data(data):
    for col in binary_features+categorical_features:
        data.loc[:, col] = pd.factorize(data[col])[0]
    data[categorical_features] = data.loc[:,categorical_features].astype('category')
    data[binary_features] = data.loc[:,binary_features].astype(int)
    data[continous_features] = data.loc[:,continous_features].astype(float)
    return data

def data_info(data):
    print("=" * 50)
    print("Dataset Information".center(50))
    print("=" * 50)

    print("\nDataset Shape:".ljust(20), data.shape)
    print("Dataset Columns:".ljust(20), list(data.columns))

    print("\n" + "-" * 50)
    print("Dataset Info".center(50))
    print("-" * 50)
    data.info()  

    print("\n" + "-" * 50)
    print("Dataset Head".center(50))
    print("-" * 50)
    print(data.head())

    print("\n" + "-" * 50)
    print("Missing Values".center(50))
    print("-" * 50)
    print(data.isnull().sum())

def descriptive_stats(data):
    print("=" * 50)
    print("Descriptive Statistics".center(50))
    print("=" * 50)

    print("\n" + "-" * 50)
    print("Summary Statistics for Continous Features".center(50))
    print("-" * 50)
    print(data[continous_features].describe())

    print("\n" + "-" * 50)
    print("Summary Statistics for Binary Features".center(50))
    print("-" * 50)

    for col in binary_features:
        counts = data[col].value_counts().sort_index()
        total = len(data)
        percentages = (counts / total * 100).round(2)
        max_class_length = max(len(str(cls)) for cls in counts.index) + 2
        print(f"\n{col}:")
        print(f"{'Class':<{max_class_length}} {'Count':<8} {'Percentage (%)':<15}")
        for cls, cnt, pct in zip(counts.index, counts, percentages):
            print(f"{cls:<{max_class_length}} {cnt:<8} {pct:<15}")

    print("\n" + "-" * 50)
    print("Summary Statistics for Categorical Features".center(50))
    print("-" * 50)

    for col in categorical_features:
        counts = data[col].value_counts().sort_index()
        total = len(data)
        percentages = (counts / total * 100).round(2)
        max_class_length = max(len(str(cls)) for cls in counts.index) + 2
        print(f"\n{col}:")
        print(f"{'Class':<{max_class_length}} {'Count':<8} {'Percentage (%)':<15}")
        for cls, cnt, pct in zip(counts.index, counts, percentages):
            print(f"{cls:<{max_class_length}} {cnt:<8} {pct:<15}")


def visualize_binary_features(data):
    for col in binary_features:
        plt.figure(figsize=(12,6))
        if col in ['gender','SeniorCitizen']:
            sns.countplot(data=data,x=col,palette='Set2',hue='Churn', stat="probability")
        else:
            sns.countplot(data=data,x=col,palette='Set2',hue='Churn', stat="probability",order=['No','Yes'])
        plt.xlabel(col)
        plt.title(f'Distribution of {col}')
        plt.ylabel("Proportion")
        plt.tight_layout()
        plt.show()

def visualize_categorical_features(data):
    for col in categorical_features:
        plt.figure(figsize=(12, 6))
        sns.countplot(x=col, hue="Churn", data=data, order=data[col].value_counts().index,palette='Set2', stat="probability")
        plt.title(f"Churn by {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend(title="Churn", labels=["No", "Yes"])
        plt.show()

def visualize_continous_features(data):
    #Histograms
    for col in continous_features:
        plt.figure(figsize=(12,6))
        sns.histplot(data[col],bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    #Boxplots
    for col in continous_features:
        plt.figure(figsize=(12,6))
        sns.boxplot(x='Churn',y=col, data=data, palette='Set2',hue='Churn')
        plt.title(f'{col} Churn vs No Churn')
        plt.xlabel('Churn')
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

    #Violinplots
    for col in continous_features:
        plt.figure(figsize=(12,6))
        sns.violinplot(x='Churn',y=col, data=data, palette='Set2',hue='Churn')
        plt.title(f'{col} Churn vs No Churn')
        plt.xlabel('Churn')
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

    #Scatterplots
    continuous_features_with_churn = continous_features + ["Churn"]  
    plt.figure(figsize=(12,6))
    g = sns.PairGrid(data[continuous_features_with_churn], hue="Churn", height=4)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
    plt.show()

def corelation_matrix(data):
    data_corelations = data
    correlation_matrix = round(data_corelations.corr('spearman'),2)
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix")
    plt.show()

def mannwhitneyu_test(data):
    for col in continous_features:
        non_churn = data[data['Churn'] == 0][col]
        churn = data[data['Churn'] == 1][col]

        u_stat, p_value = mannwhitneyu(non_churn, churn)  

        print(f"Mann-Whitney U Test Results for {col}:")
        print(f"U-statistic: {u_stat:.4f}")
        print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"The distribution of {col} differs significantly between churn and non-churn (p < 0.05).")
    else:
        print(f"No significant difference in {col} distribution between churn and non-churn (p >= 0.05).")
    print()

def chi_squred_test(data):
    for col in binary_features+categorical_features:
        contingency_table = pd.crosstab(data[col], data["Churn"])
        chi2_stat, p_value, dof, _ = chi2_contingency(contingency_table)
        if col!='Churn':
            print(f"Chi-Squared Test Results for {col}:")
            print(f"Chi-Squared Statistic: {chi2_stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Degrees of Freedom: {dof}")
            
            if p_value < 0.05:
                print(f"The relationship between {col} and Churn is statistically significant (p < 0.05).")
            else:
                print(f"The relationship between {col} and Churn is not statistically significant (p >= 0.05).")
            print()  