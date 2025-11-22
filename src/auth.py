import sqlite3
import bcrypt
import streamlit as st
from pathlib import Path

DB_PATH = Path("users.db")

def init_db():
    """Initialize the SQLite database for users."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS users
           (email TEXT PRIMARY KEY, password_hash BLOB)'''
    )
    conn.commit()
    conn.close()

def create_user(email, password):
    """Create a new user with a hashed password."""
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, password_hash))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(email, password):
    """Verify user credentials."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    conn.close()
    
    if result:
        stored_hash = result[0]
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
    return False

def check_authentication():
    """
    Handles manual authentication (Sign In / Sign Up).
    Returns the user's email if authenticated, None otherwise.
    """
    # Initialize DB on first run
    if not DB_PATH.exists():
        init_db()

    if "email" not in st.session_state:
        st.session_state.email = None

    if st.session_state.email:
        return st.session_state.email

    st.header("Welcome")
    
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Sign In")
            
            if submit:
                if verify_user(email, password):
                    st.session_state.email = email
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid email or password")

    with tab2:
        with st.form("signup_form"):
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
            submit_signup = st.form_submit_button("Sign Up")
            
            if submit_signup:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    if create_user(new_email, new_password):
                        st.success("Account created! You can now sign in.")
                    else:
                        st.error("Email already registered")

    return None

def logout():
    """Logs the user out."""
    st.session_state.email = None
    st.rerun()
