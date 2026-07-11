import os
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'DatasetAfricaMalaria.csv')
    model_path = os.path.join(base_dir, 'models', 'risk_rf_model.pkl')
    shap_path = os.path.join(base_dir, 'models', 'metrics', 'risk_shap_plot.png')
    
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
    
    df_clean = df.dropna(subset=[target_col] + feature_cols)
    labels = ["Low", "Medium", "High"]
    df_clean['Risk_Level'] = pd.qcut(df_clean[target_col], q=3, labels=labels)
    
    X = df_clean[feature_cols]
    y = df_clean['Risk_Level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    with open(model_path, 'rb') as f:
        rf = pickle.load(f)
        
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure()
    # Use plot_size to ensure it has enough room
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, plot_size=(14, 8))
    
    # Adjust layout so long labels fit without squishing the chart
    plt.title('SHAP Feature Importance', pad=20)
    plt.tight_layout()
    
    plt.savefig(shap_path, bbox_inches='tight', dpi=150)
    plt.close()
    print("SHAP plot regenerated successfully.")

if __name__ == "__main__":
    main()
