import streamlit as st
import os

st.set_page_config(page_title="Malaria Assessment Framework", layout="wide")

# Initialize session state for mock auth
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "user_name" not in st.session_state:
    st.session_state["user_name"] = None

from ui.auth_ui import render_auth
from ui.diagnosis_ui import render_diagnosis
from ui.risk_ui import render_risk
from ui.combined_ui import render_combined
from src.database.db import init_db

def main():
    # Initialize the database on startup (crucial for cloud hosting!)
    init_db()
    
    st.title("Malaria Diagnosis and Risk Forecasting Framework")
    st.markdown("An Explainable, Multimodal AI Framework")

    if st.session_state["user_id"] is None:
        render_auth()
    else:
        st.sidebar.markdown(f"**Logged in as:** {st.session_state['user_name']}")
        if st.sidebar.button("Logout"):
            st.session_state["user_id"] = None
            st.session_state["user_name"] = None
            st.rerun()

        tabs = st.tabs(["1. Image Diagnosis", "2. Risk Estimation", "3. Combined Assessment"])
        
        with tabs[0]:
            render_diagnosis()
            
        with tabs[1]:
            render_risk()
            
        with tabs[2]:
            render_combined()

if __name__ == "__main__":
    main()
