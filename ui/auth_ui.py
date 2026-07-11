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
