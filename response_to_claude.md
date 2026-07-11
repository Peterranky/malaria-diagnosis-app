# Response to Claude (Project Documentation)

## 1. Authentication System Confirmation
Regarding your question on the authentication mechanism:
**It is a mock stub.** The application writes to the SQLite `users` table, but passwords are currently stored in plain text. The column is named `password_hash` to reflect a production-ready schema design, but for this prototype, `app.py` explicitly notes that this is 'mock auth'. To claim cryptographic security in the documentation would be overclaiming.

## 2. Project Directory Tree
```text
Malaria Diagnosis and risk forecasting/
    app.py
    compile_response.py
    DatasetAfricaMalaria.csv
    fix_metrics.py
    fix_shap.py
    malaria_framework.db
    requirements.txt
    temp_cam.jpg
    temp_upload.jpg
    models/
        image_model.keras
        risk_lr_model.pkl
        risk_rf_model.pkl
        metrics/
            image_confusion_matrix.png
            image_metrics.json
            risk_confusion_matrix.png
            risk_metrics.json
            risk_shap_plot.png
    src/
        database/
            db.py
        image_pipeline/
            inference.py
            train.py
        risk_pipeline/
            train.py
    ui/
        auth_ui.py
        combined_ui.py
        diagnosis_ui.py
        risk_ui.py
```

## 3. Source Code Files (Appendix A)

### ui/auth_ui.py
```python
import streamlit as st
from src.database.db import get_connection

def render_auth():
    st.subheader("Login or Register")
    auth_mode = st.radio("Mode", ["Login", "Register"])
    
    with st.form("auth_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        name = ""
        if auth_mode == "Register":
            name = st.text_input("Name")
            
        submitted = st.form_submit_button("Submit")
        if submitted:
            if auth_mode == "Register":
                if email and password and name:
                    try:
                        conn = get_connection()
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)", (name, email, password))
                        conn.commit()
                        conn.close()
                        st.success("Registered successfully! Please log in.")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Please fill all fields.")
            else:
                if email and password:
                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT user_id, name FROM users WHERE email=? AND password_hash=?", (email, password))
                    user = cursor.fetchone()
                    conn.close()
                    if user:
                        st.session_state["user_id"] = user[0]
                        st.session_state["user_name"] = user[1]
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
                else:
                    st.warning("Please fill all fields.")

```

### ui/diagnosis_ui.py
```python
import streamlit as st
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from src.database.db import get_connection
from src.image_pipeline.inference import make_gradcam_heatmap, save_and_display_gradcam

@st.cache_resource
def load_image_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'image_model.keras')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

def render_diagnosis():
    st.header("Image Diagnosis")
    st.markdown("Upload a single-cell blood smear image to detect the presence of malaria parasites.")
    
    # Display metrics
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'metrics', 'image_metrics.json')
    cm_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'metrics', 'image_confusion_matrix.png')
    
    with st.expander("Model Performance Evidence (Test Set)"):
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            st.write(f"**Accuracy:** {metrics.get('accuracy', 0):.2%}")
            st.write(f"**Precision:** {metrics.get('precision', 0):.2%}")
            st.write(f"**Recall / Sensitivity:** {metrics.get('recall', 0):.2%}")
            st.write(f"**Specificity:** {metrics.get('specificity', 0):.2%}")
            st.write(f"**F1 Score:** {metrics.get('f1_score', 0):.2%}")
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix")
        else:
            st.warning("Model metrics not found. Please train the model first.")
            
    model = load_image_model()
    if model is None:
        st.error("Image model not found. Please ensure the model is trained.")
        return
        
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption='Uploaded Image', use_container_width=True)
            
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                # Save uploaded file temporarily for Grad-CAM processing
                temp_path = "temp_upload.jpg"
                img.convert('RGB').save(temp_path)
                
                # Preprocess for prediction
                img_array = tf.keras.preprocessing.image.load_img(temp_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img_array)
                img_array = (img_array / 127.5) - 1.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                preds = model.predict(img_array)
                prob = float(preds[0][0])
                
                prediction = "Uninfected" if prob > 0.5 else "Parasitized"
                confidence = prob if prob > 0.5 else 1.0 - prob
                
                st.success(f"**Prediction:** {prediction}")
                st.info(f"**Confidence:** {confidence:.2%}")
                
                # Grad-CAM
                st.write("Generating Grad-CAM heatmap...")
                heatmap = make_gradcam_heatmap(img_array, model)
                cam_path = "temp_cam.jpg"
                save_and_display_gradcam(temp_path, heatmap, cam_path)
                
                with col2:
                    st.image(cam_path, caption='Grad-CAM Heatmap', use_container_width=True)
                    
                # Save to database
                if st.session_state.get("user_id"):
                    try:
                        conn = get_connection()
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT INTO diagnosis_records (user_id, image_path, gradcam_path, prediction, confidence) VALUES (?, ?, ?, ?, ?)",
                            (st.session_state["user_id"], temp_path, cam_path, prediction, confidence)
                        )
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        st.error(f"Error saving record: {e}")

```

### ui/risk_ui.py
```python
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
            
            st.success(f"**Predicted Risk Level:** {prediction}")
            
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

```

### ui/combined_ui.py
```python
import streamlit as st
import pandas as pd
from src.database.db import get_connection
import os

def render_combined():
    st.header("Combined Assessment")
    st.markdown("Decision-level integration of Image Diagnosis and Risk Estimation.")
    
    if not st.session_state.get("user_id"):
        st.warning("Please log in to view combined assessments.")
        return
        
    try:
        conn = get_connection()
        
        # Get latest image diagnosis for user
        diag_query = "SELECT prediction, confidence, created_at FROM diagnosis_records WHERE user_id=? ORDER BY created_at DESC LIMIT 1"
        diag_df = pd.read_sql(diag_query, conn, params=(st.session_state["user_id"],))
        
        # Get latest risk estimation for user
        risk_query = "SELECT risk_level, country, year, created_at FROM risk_records WHERE user_id=? ORDER BY created_at DESC LIMIT 1"
        risk_df = pd.read_sql(risk_query, conn, params=(st.session_state["user_id"],))
        
        conn.close()
        
        col1, col2 = st.columns(2)
        
        has_diag = not diag_df.empty
        has_risk = not risk_df.empty
        
        with col1:
            st.subheader("Latest Image Diagnosis")
            if has_diag:
                diag = diag_df.iloc[0]
                st.write(f"**Prediction:** {diag['prediction']}")
                st.write(f"**Confidence:** {diag['confidence']:.2%}")
                st.write(f"*(Date: {diag['created_at']})*")
            else:
                st.write("No image diagnosis found.")
                
        with col2:
            st.subheader("Latest Risk Profile")
            if has_risk:
                risk = risk_df.iloc[0]
                st.write(f"**Risk Level:** {risk['risk_level']}")
                st.write(f"**Context:** {risk['country']} ({risk['year']})")
                st.write(f"*(Date: {risk['created_at']})*")
            else:
                st.write("No risk profile found.")
                
        if has_diag and has_risk:
            st.markdown("---")
            st.subheader("Combined Clinical Context")
            
            diag_res = diag['prediction']
            risk_res = risk['risk_level']
            
            st.info(f"**Rule-Based Assessment Logic:** Image = {diag_res} | Regional Risk = {risk_res}")
            
            if diag_res == "Parasitized":
                if risk_res == "High":
                    st.error("⚠️ **High Alert:** The image confirms the presence of malaria parasites, and the regional context indicates a High-Risk environment for incidence. Immediate standard antimalarial protocols should be considered, factoring in local drug-resistance patterns.")
                elif risk_res == "Medium":
                    st.warning("⚠️ **Alert:** The image confirms malaria parasites. The regional context is Medium-Risk. Follow standard treatment protocols.")
                else:
                    st.warning("⚠️ **Alert:** The image confirms malaria parasites, despite the regional context being Low-Risk. Confirm recent travel history and follow standard treatment protocols.")
            else:
                if risk_res == "High":
                    st.warning("ℹ️ **Observation:** The image shows No Parasites (Uninfected). However, the patient is in a High-Risk environment. Ensure symptoms are monitored closely or consider alternative diagnoses (e.g., typhoid, dengue) if fever persists.")
                else:
                    st.success("✅ **Clear:** The image is clear of parasites, and the regional risk is Low/Medium. Malaria is unlikely based on this combination.")
                    
    except Exception as e:
        st.error(f"Error fetching data: {e}")

```

### src/risk_pipeline/inference.py
```python
# Error reading file: [Errno 2] No such file or directory: 'C:\\Users\\LENOVO\\OneDrive\\Documents\\Malaria Diagnosis and risk forecasting\\src\\risk_pipeline\\inference.py'
```

### src/image_pipeline/train.py (Snippet - Metrics Fix)
```python
    # Treating 0 as the positive class for sensitivity/specificity calculations if we want to predict "Parasitized"
    # But let's just use sklearn metrics natively
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[0, 0]   # parasitized correctly detected
    fn = cm[0, 1]   # parasitized missed -> false negatives
    fp = cm[1, 0]   # uninfected flagged as parasitized
    tn = cm[1, 1]   # uninfected correctly cleared

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall      = sensitivity
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy    = (tp + tn) / (tp + tn + fp + fn)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity)

```

