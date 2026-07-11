import streamlit as st
import os
import json
import pickle
import pandas as pd
from src.database.db import get_connection

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'DatasetAfricaMalaria.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return pd.DataFrame()

@st.cache_resource
def load_risk_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'risk_rf_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

def render_risk():
    st.header("Malaria Risk Estimation")
    st.markdown("Estimate malaria incidence risk based on World Bank indicator data (WASH, urbanization).")
    
    # Evidence metrics
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'metrics', 'risk_metrics.json')
    cm_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'metrics', 'risk_confusion_matrix.png')
    shap_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'metrics', 'risk_shap_plot.png')
    
    with st.expander("Model Performance Evidence (Test Set)"):
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            st.write(f"**Accuracy:** {metrics.get('accuracy', 0):.2%}")
            st.write(f"**Precision:** {metrics.get('precision', 0):.2%}")
            st.write(f"**Recall:** {metrics.get('recall', 0):.2%}")
            st.write(f"**F1 Score:** {metrics.get('f1_score', 0):.2%}")
            
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Confusion Matrix (Low/Medium/High)")
            with col2:
                if os.path.exists(shap_path):
                    st.image(shap_path, caption="SHAP Feature Importance")
        else:
            st.warning("Model metrics not found. Please train the risk model first.")
            
    model = load_risk_model()
    df = load_data()
    
    if model is None or df.empty:
        st.error("Risk model or dataset not found. Please ensure the pipeline has run.")
        return
        
    st.subheader("Predict Risk")
    
    # Filter Nigeria or all
    filter_nigeria = st.checkbox("Filter to Nigeria only", value=False)
    
    countries = sorted(df['Country Name'].dropna().unique())
    if filter_nigeria:
        countries = [c for c in countries if "Nigeria" in c]
        
    selected_country = st.selectbox("Select Country", countries)
    
    country_data = df[df['Country Name'] == selected_country]
    years = sorted(country_data['Year'].dropna().unique())
    selected_year = st.selectbox("Select Year", years)
    
    row_data = country_data[country_data['Year'] == selected_year]
    
    if not row_data.empty:
        row = row_data.iloc[0]
        actual_incidence = row.get("Incidence of malaria (per 1,000 population at risk)", "N/A")
        st.write(f"**Actual historical incidence (per 1,000):** {actual_incidence}")
        
        st.markdown("### Indicator Values (Manual What-If Entry)")
        
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
        
        input_data = {}
        for feature in feature_cols:
            val = row.get(feature, 0.0)
            if pd.isna(val):
                val = 0.0
            input_data[feature] = st.number_input(feature, value=float(val))
            
        if st.button("Estimate Risk"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            
            # Show predict_proba as confidence
            proba = model.predict_proba(input_df)[0]
            # model.classes_ might be ["Low", "Medium", "High"] but we find the proba for the predicted class
            predicted_class_idx = list(model.classes_).index(prediction)
            confidence = proba[predicted_class_idx]
            
            st.success(f"**Predicted Risk Level:** {prediction}")
            st.info(f"**Confidence:** {confidence:.2%}")
            
            # Simple rule-based explanation
            st.info("**Top driver insight:** The model's global SHAP values show that rural population percentage and basic sanitation access heavily influence this result.")
            
            if st.session_state.get("user_id"):
                try:
                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO risk_records (user_id, country, year, water_basic, sanitation_basic, urban_pop_pct, risk_level, explanation)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        st.session_state["user_id"],
                        selected_country,
                        int(selected_year),
                        input_data.get("People using at least basic drinking water services (% of population)", 0),
                        input_data.get("People using at least basic sanitation services (% of population)", 0),
                        input_data.get("Urban population (% of total population)", 0),
                        prediction,
                        "Based on inputted WASH and urbanization features."
                    ))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    st.error(f"Error saving record: {e}")
    else:
        st.warning("No data found for this selection.")
