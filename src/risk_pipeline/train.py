import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'DatasetAfricaMalaria.csv')
    df = pd.read_csv(data_path)
    
    target_col = "Incidence of malaria (per 1,000 population at risk)"
    feature_cols = [
        "Year",
        "Rural population (% of total population)",
        "Rural population growth (annual %)",
        "Urban population (% of total population)",
        "Urban population growth (annual %)",
        "People using at least basic drinking water services (% of population)",
        "People using at least basic drinking water services, rural (% of rural population)",
        "People using at least basic drinking water services, urban (% of urban population)",
        "People using at least basic sanitation services (% of population)",
        "People using at least basic sanitation services, rural (% of rural population)",
        "People using at least basic sanitation services, urban  (% of urban population)"
    ]
    
    # Drop rows missing target or key features
    df_clean = df.dropna(subset=[target_col] + feature_cols)
    print(f"Original rows: {len(df)}, Rows after dropping missing values: {len(df_clean)}")
    
    # Bin the target into tertiles
    labels = ["Low", "Medium", "High"]
    df_clean['Risk_Level'] = pd.qcut(df_clean[target_col], q=3, labels=labels)
    
    X = df_clean[feature_cols]
    y = df_clean['Risk_Level']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    with open(os.path.join(models_dir, 'risk_lr_model.pkl'), 'wb') as f:
        pickle.dump(lr, f)
    with open(os.path.join(models_dir, 'risk_rf_model.pkl'), 'wb') as f:
        pickle.dump(rf, f)
        
    print("Models saved.")
    
    # Evaluate RF (Best model usually)
    y_pred = rf.predict(X_test)
    
    # Ensure metrics match requirements
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1)
    }
    
    metrics_dir = os.path.join(models_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, 'risk_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Metrics saved.")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Risk Estimation Confusion Matrix')
    cm_path = os.path.join(metrics_dir, 'risk_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    print("Confusion matrix saved.")
    
    # SHAP
    explainer = shap.TreeExplainer(rf)
    # SHAP returns a list of arrays for multiclass
    shap_values = explainer.shap_values(X_test)
    
    # Plot SHAP summary for class 2 (High risk) for example, or a general feature importance
    # We will plot feature importance via SHAP magnitude across all classes
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance')
    shap_path = os.path.join(metrics_dir, 'risk_shap_plot.png')
    plt.savefig(shap_path, bbox_inches='tight')
    plt.close()
    
    print("SHAP plot saved.")

if __name__ == "__main__":
    main()
