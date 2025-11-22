import os
import streamlit as st
from streamlit_oauth import OAuth2Component
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Constants
CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("REDIRECT_URI", "http://localhost:8501")
AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_TOKEN_URL = "https://oauth2.googleapis.com/revoke"
SCOPE = "openid email profile"


def check_authentication():
    """
    Handles Google OAuth authentication.
    Returns the user's email if authenticated, None otherwise.
    """
    if "email" not in st.session_state:
        st.session_state.email = None

    if st.session_state.email:
        return st.session_state.email

    # If not authenticated, show login button
    st.header("Sign In")
    st.caption("Please sign in with your Google account to access the dashboard.")

    if not CLIENT_ID or not CLIENT_SECRET:
        st.error("Missing Google OAuth credentials. Please configure GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.")
        return None

    oauth2 = OAuth2Component(
        CLIENT_ID, CLIENT_SECRET, AUTHORIZATION_URL, TOKEN_URL, TOKEN_URL, REVOKE_TOKEN_URL
    )

    result = oauth2.authorize_button(
        name="Continue with Google",
        icon="https://www.google.com.tw/favicon.ico",
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        key="google_auth",
        extras_params={"prompt": "consent", "access_type": "offline"},
    )

    if result:
        # Decode the ID token to get user info
        # Note: In a production app, you should verify the signature of the JWT.
        # streamlit-oauth returns the token response.
        try:
            # Simple decoding for demonstration. 
            # Ideally use google-auth library to verify token.
            import jwt
            id_token = result.get("token", {}).get("id_token")
            if id_token:
                decoded = jwt.decode(id_token, options={"verify_signature": False})
                email = decoded.get("email")
                st.session_state.email = email
                st.session_state.token = result.get("token")
                st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}")
    
    return None


def logout():
    """Logs the user out."""
    if "email" in st.session_state:
        del st.session_state.email
    if "token" in st.session_state:
        del st.session_state.token
    st.rerun()
