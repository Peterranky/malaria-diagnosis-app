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
                st.progress(float(confidence))
                
                if 0.40 <= prob <= 0.60:
                    st.warning("Low confidence - expert microscopy review recommended")
                
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
