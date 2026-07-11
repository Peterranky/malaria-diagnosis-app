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
            st.subheader("Image Diagnosis (Grad-CAM)")
            if has_diag:
                diag = diag_df.iloc[0]
                st.write(f"**Prediction:** {diag['prediction']}")
                st.write(f"**Confidence:** {diag['confidence']:.2%}")
                st.write(f"*(Date: {diag['created_at']})*")
            else:
                st.write("No image diagnosis found.")
                
        with col2:
            st.subheader("Environmental Risk (SHAP)")
            if has_risk:
                risk = risk_df.iloc[0]
                st.write(f"**Risk Level:** {risk['risk_level']}")
                st.write(f"**Context:** {risk['country']} ({risk['year']})")
                st.write(f"*(Date: {risk['created_at']})*")
            else:
                st.write("No risk profile found.")
                
        if has_diag and has_risk:
            st.markdown("---")
            st.subheader("Decision-Level Fusion")
            
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
