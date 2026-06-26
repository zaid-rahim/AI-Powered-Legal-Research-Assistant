import streamlit as st
from backend_wrapper import ask_question
from pyvis.network import Network
import streamlit.components.v1 as components
import re, time, html as html_lib
import json, os, hashlib


# AUTH HELPERS

# This section handles basic, file-based user authentication.
# Usernames and hashed passwords are saved in a local JSON file (users.json).
# WARNING: This is for prototyping. Do not use plain JSON for auth in production.
USERS_FILE = "users.json"


def _hash_pw(password: str) -> str:
    """
    Creates a secure SHA-256 hash of the password combined with a static salt.
    Salting prevents dictionary attacks.
    """
    return hashlib.sha256(("lawpak_salt_" + password).encode()).hexdigest()


def _load_users() -> dict:
    """
    Loads the user database from the local JSON file.
    Returns an empty dictionary if the file doesn't exist or is corrupted.
    """
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_users(users: dict):
    """
    Writes the updated user dictionary back to the local JSON file.
    """
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _register_user(username: str, password: str) -> tuple[bool, str]:
    """
    Validates credentials and registers a new user if the username is unique.
    
    Returns:
        tuple (bool, str): (Success status, Status message)
    """
    uname = username.strip().lower()
    if len(uname) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    users = _load_users()
    if uname in users:
        return False, "Username already taken."
    users[uname] = {"password_hash": _hash_pw(password)}
    _save_users(users)
    return True, "Account created successfully."


def _verify_user(username: str, password: str) -> tuple[bool, str]:
    """
    Checks if the provided username exists and if the hashed password matches.
    
    Returns:
        tuple (bool, str): (Success status, Status message)
    """
    uname = username.strip().lower()
    users = _load_users()
    if uname not in users:
        return False, "User not found."
    if users[uname]["password_hash"] != _hash_pw(password):
        return False, "Incorrect password."
    return True, "Login successful."



# PAGE CONFIG

# Must be the first Streamlit command. Sets the browser tab title and layout width.
st.set_page_config(
    page_title="LAWPAK – Legal Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


# SESSION STATE

# Streamlit reruns the script top-to-bottom on every user interaction.
# st.session_state is used to persist variables (like login status and chat history)
# across these reruns.

# Initialize core application states
for _k, _v in {
    "user_authenticated": False,
    "current_user": None,
    "auth_page": "login",
    "current_page": "chat",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

for _k in ["messages", "show_sources", "show_summary", "show_kg"]:
    if _k not in st.session_state:
        st.session_state[_k] = []


# 
# AUTH GATE
# 
# If the user is not logged in, we intercept the execution flow here.
# We display the Login/Signup screen and use `st.stop()` to prevent the rest
# of the application from rendering until authentication succeeds.
if not st.session_state.user_authenticated:

    # Determine if we should show the signup or login view
    _signup = st.session_state.auth_page == "signup"
    _bg = "#f0f2f0" if _signup else "#0f1a10"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Merriweather:wght@700&display=swap');
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ font-family: 'Inter', sans-serif; height: 100%; }}

#MainMenu, footer,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stHeader"],
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{
    display: none !important;
}}

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    background: {_bg} !important;
    min-height: 100vh !important;
}}
[data-testid="stMainBlockContainer"] {{
    max-width: 440px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
}}
[data-testid="stVerticalBlock"] {{ gap: 0 !important; }}

.auth-spacer {{ height: clamp(28px, 5vh, 56px); }}
.auth-logo-wrap {{
    text-align: center; margin-bottom: 8px;
}}
.auth-logo {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 46px; height: 46px; background: #111;
    border-radius: 50%; font-size: 22px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
}}
.auth-brand-name {{
    font-family: 'Merriweather', serif;
    font-size: 17px; font-weight: 700; color: {'#111' if _signup else '#fff'};
    text-align: center; margin: 6px 0 2px;
}}
.auth-brand-sub {{
    font-size: 11.5px; color: {'#888' if _signup else 'rgba(255,255,255,0.45)'};
    text-align: center; margin-bottom: 18px;
}}

.auth-card {{
    background: #ffffff;
    border-radius: 18px;
    padding: 28px 30px 24px;
    box-shadow: {'0 4px 24px rgba(0,0,0,0.08)' if _signup else '0 16px 56px rgba(0,0,0,0.5)'};
    margin-bottom: 10px;
}}
.auth-card-title {{
    font-size: 15px; font-weight: 700; color: #111; margin-bottom: 2px;
}}
.auth-card-sub {{
    font-size: 11.5px; color: #aaa; margin-bottom: 16px;
}}

[data-testid="stTextInput"] label {{
    font-size: 12.5px !important; font-weight: 500 !important; color: #444 !important;
}}
[data-testid="stTextInput"] input {{
    border: 1.5px solid #e0e0e0 !important; border-radius: 9px !important;
    padding: 9px 12px !important; font-size: 13.5px !important;
    background: #fff !important; color: #111 !important;
    -webkit-text-fill-color: #111 !important;
    transition: border-color .2s !important;
}}
[data-testid="stTextInput"] input:focus {{
    border-color: #111 !important;
    box-shadow: 0 0 0 3px rgba(0,0,0,0.06) !important;
}}
[data-testid="stFormSubmitButton"] > button {{
    background: #111 !important; color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    border: none !important; border-radius: 9px !important;
    padding: 10px 0 !important; width: 100% !important;
    font-size: 13.5px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important; margin-top: 6px !important;
    cursor: pointer !important; transition: background .15s !important;
}}
[data-testid="stFormSubmitButton"] > button:hover {{ background: #333 !important; }}

.auth-footer {{
    font-size: 10.5px; color: #ccc; text-align: center; margin-top: 10px;
    line-height: 1.5;
}}
.auth-footer a {{ color: #999; text-decoration: underline; }}

[data-testid="stButton"] > button {{
    background: transparent !important;
    color: {'#555' if _signup else 'rgba(255,255,255,0.65)'} !important;
    -webkit-text-fill-color: {'#555' if _signup else 'rgba(255,255,255,0.65)'} !important;
    border: 1.5px solid {'#e0e0e0' if _signup else 'rgba(255,255,255,0.18)'} !important;
    border-radius: 9px !important; width: 100% !important;
    padding: 9px 0 !important; font-size: 13px !important; font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important; transition: all .15s !important;
}}
[data-testid="stButton"] > button:hover {{
    background: {'#f5f5f5' if _signup else 'rgba(255,255,255,0.08)'} !important;
    color: {'#111' if _signup else '#fff'} !important;
    -webkit-text-fill-color: {'#111' if _signup else '#fff'} !important;
    border-color: {'#ccc' if _signup else 'rgba(255,255,255,0.35)'} !important;
}}
[data-testid="stAlert"] {{
    border-radius: 9px !important; font-size: 12.5px !important; margin-top: 6px !important;
}}
</style>
""", unsafe_allow_html=True)

    # Top spacer
    st.markdown('<div class="auth-spacer"></div>', unsafe_allow_html=True)

    # Brand
    st.markdown('<div class="auth-logo-wrap"><div class="auth-logo">⚖️</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-brand-name">LAWPAK</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-brand-sub">Pakistan Legal Research Assistant</div>', unsafe_allow_html=True)

    # Card
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)

    if _signup:
        st.markdown('<div class="auth-card-title">Create an account</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-card-sub">Fill in the details below to get started</div>', unsafe_allow_html=True)

        with st.form("signup_form", clear_on_submit=False):
            su_user = st.text_input("Username", placeholder="Choose a username")
            su_pass = st.text_input("Password", type="password", placeholder="Min 4 characters")
            su_pass2 = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
            su_submit = st.form_submit_button("Create Account", use_container_width=True)

        if su_submit:
            if not su_user.strip():
                st.error("Please enter a username.")
            elif su_pass != su_pass2:
                st.error("Passwords do not match.")
            else:
                ok, msg = _register_user(su_user, su_pass)
                if ok:
                    st.success(msg)
                    st.session_state.user_authenticated = True
                    st.session_state.current_user = su_user.strip().lower()
                    time.sleep(0.4)
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown('<p class="auth-footer">By creating an account you agree to our <a href="#">Terms of use</a> and <a href="#">Privacy Policy</a>.</p>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="auth-card-title">Sign in</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-card-sub">Welcome back — enter your credentials to continue</div>', unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            li_user = st.text_input("Username", placeholder="Enter your username")
            li_pass = st.text_input("Password", type="password", placeholder="Enter your password")
            li_submit = st.form_submit_button("Sign In", use_container_width=True)

        if li_submit:
            if not li_user.strip():
                st.error("Please enter a username.")
            elif not li_pass:
                st.error("Please enter your password.")
            else:
                ok, msg = _verify_user(li_user, li_pass)
                if ok:
                    st.session_state.user_authenticated = True
                    st.session_state.current_user = li_user.strip().lower()
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown('<p class="auth-footer">Protected by reCAPTCHA — <a href="#">Learn more</a>.</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close .auth-card

    # Toggle button
    if _signup:
        if st.button("Already have an account? Sign in →", use_container_width=True):
            st.session_state.auth_page = "login"
            st.rerun()
    else:
        if st.button("Don't have an account? Sign up →", use_container_width=True):
            st.session_state.auth_page = "signup"
            st.rerun()

    st.stop()


# 
# MAIN APPLICATION
# 
# If execution reaches this point, the user is authenticated.

# Inject custom CSS to completely override Streamlit's default styling,
# giving the application a custom, premium, full-width chat interface.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Merriweather:wght@700&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body { font-family: 'Inter', sans-serif; overflow-x: hidden; background: #fff; }

#MainMenu, footer,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* Keep the header transparent but preserve sidebar toggle button */
[data-testid="stHeader"] {
    background: transparent !important;
    border: none !important; box-shadow: none !important;
    z-index: 999 !important; pointer-events: none;
}
[data-testid="stHeader"] button,
[data-testid="stHeader"] [data-testid="collapsedControl"] {
    pointer-events: all !important;
}

/* Sidebar collapse/expand pill — floats outside sidebar */
[data-testid="collapsedControl"] {
    background: #111 !important;
    border: 1px solid #333 !important;
    border-radius: 0 8px 8px 0 !important;
    color: #fff !important;
    left: 0 !important; top: 14px !important;
    padding: 7px 9px !important; z-index: 1001 !important;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.25) !important;
    transition: background .2s !important;
}
[data-testid="collapsedControl"]:hover {
    background: #333 !important; border-color: #555 !important;
}
[data-testid="collapsedControl"] svg {
    fill: #fff !important; stroke: #fff !important;
    width: 16px !important; height: 16px !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"] { background: #fff !important; }
[data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }

/* ── SIDEBAR ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #111 !important;
    border-right: 1px solid #1e1e1e !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #111 !important; padding: 0 !important;
    height: 100vh !important; overflow-y: auto !important;
}
[data-testid="stSidebar"] h3 {
    font-family: 'Merriweather', serif !important;
    font-size: 16px !important; font-weight: 700 !important;
    color: #fff !important; -webkit-text-fill-color: #fff !important;
    padding: 18px 16px 3px !important; margin: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaption"] p {
    color: #555 !important; -webkit-text-fill-color: #555 !important;
    font-size: 11px !important; padding: 0 16px 8px !important; margin: 0 !important;
}
[data-testid="stSidebar"] hr {
    border: none !important; border-top: 1px solid #252525 !important;
    margin: 4px 12px !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 12px !important; color: #b0b0b0 !important;
    -webkit-text-fill-color: #b0b0b0 !important;
    padding: 5px 14px !important; margin: 1px 5px !important;
    border-radius: 7px !important; line-height: 1.4 !important;
    transition: background .12s !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p:hover {
    background: #1e1e1e !important; color: #fff !important;
    -webkit-text-fill-color: #fff !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
    font-size: 9.5px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: .12em !important;
    color: #444 !important; -webkit-text-fill-color: #444 !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background: #1a1a1a !important; color: #d0d0d0 !important;
    -webkit-text-fill-color: #d0d0d0 !important;
    border: 1px solid #2a2a2a !important; border-radius: 8px !important;
    width: 100% !important; padding: 7px 14px !important;
    font-size: 12.5px !important; font-weight: 500 !important;
    text-align: left !important; font-family: 'Inter', sans-serif !important;
    margin: 1px 0 !important; transition: background .12s !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: #252525 !important; color: #fff !important;
    -webkit-text-fill-color: #fff !important; border-color: #333 !important;
}
[data-testid="stSidebar"] button[kind="header"],
[data-testid="stSidebar"] [data-testid="baseButton-header"] {
    color: #aaa !important; -webkit-text-fill-color: #aaa !important;
}
[data-testid="stSidebar"] button[kind="header"]:hover,
[data-testid="stSidebar"] [data-testid="baseButton-header"]:hover {
    color: #fff !important; -webkit-text-fill-color: #fff !important;
}
.logout-btn button {
    background: #2a1a1a !important; border-color: #3a2222 !important;
    color: #e88 !important; -webkit-text-fill-color: #e88 !important;
}
.logout-btn button:hover {
    background: #3a2222 !important; color: #f99 !important;
    -webkit-text-fill-color: #f99 !important;
}

/* ── TOP BAR ────────────────────────────────────────────────────── */
.topbar {
    position: sticky; top: 0; z-index: 90;
    background: rgba(255,255,255,0.96);
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border-bottom: 1px solid #ebebeb;
    padding: 10px 28px;
    display: flex; align-items: center; justify-content: space-between;
}
.topbar-title { font-family: 'Merriweather', serif; font-size: 14px; font-weight: 700; color: #111; }
.topbar-sub { font-size: 11px; color: #aaa; margin-top: 1px; }
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #111; color: #fff; font-size: 10.5px; font-weight: 500;
    padding: 4px 12px; border-radius: 100px; white-space: nowrap;
}
.badge-dot {
    width: 6px; height: 6px; border-radius: 50%; background: #22c55e;
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ── CONTENT AREA ────────────────────────────────────────────────── */
.content-wrap {
    max-width: 780px; margin: 0 auto; width: 100%;
    padding: 0 20px 160px;
}

/* ── WELCOME ─────────────────────────────────────────────────────── */
.welcome-hero {
    display: flex; flex-direction: column; align-items: center;
    text-align: center; padding: 40px 12px 24px;
}
.welcome-icon {
    width: 52px; height: 52px; background: #111; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; margin-bottom: 16px;
}
.welcome-hero h1 {
    font-family: 'Merriweather', serif;
    font-size: clamp(18px, 2.3vw, 26px);
    font-weight: 700; color: #111; margin-bottom: 10px; line-height: 1.35;
}
.welcome-hero p {
    font-size: 13.5px; color: #888; line-height: 1.65; max-width: 380px;
}
.cards-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 12px; margin-top: 28px; width: 100%; max-width: 560px;
}
.feat-card {
    background: #fafafa; border: 1px solid #e8e8e8; border-radius: 14px;
    padding: 18px 16px; text-align: left; transition: all .18s;
}
.feat-card:hover { background: #f3f3f3; border-color: #d0d0d0; transform: translateY(-1px); }
.feat-icon { font-size: 22px; margin-bottom: 10px; }
.feat-title { font-size: 13px; font-weight: 600; color: #111; margin-bottom: 5px; }
.feat-desc { font-size: 11.5px; color: #999; line-height: 1.5; }

/* ── CHAT MESSAGES ───────────────────────────────────────────────── */
.user-msg { display: flex; justify-content: flex-end; margin: 14px 0 4px; animation: rise .2s ease; }
.user-bubble {
    background: #111; color: #fff; padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 68%; font-size: 13.5px; line-height: 1.6; word-break: break-word;
}
.bot-row { display: flex; gap: 10px; align-items: flex-start; margin: 14px 0 4px; animation: rise .2s ease; }
.bot-av {
    width: 30px; height: 30px; flex-shrink: 0; background: #111; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; color: #fff; font-size: 13px;
}
.bot-bubble {
    flex: 1; border: 1px solid #e8e8e8; padding: 12px 16px;
    border-radius: 4px 18px 18px 18px; background: #fff;
    font-size: 13.5px; color: #111; line-height: 1.78;
    box-shadow: 0 1px 5px rgba(0,0,0,.04); word-break: break-word;
}
.bot-bubble p, .bot-bubble li { color: #111 !important; }
.bot-bubble ul, .bot-bubble ol { padding-left: 18px; }
.bot-bubble li { margin-bottom: 3px; }
.bot-bubble h3 {
    font-size: 13.5px; font-weight: 700; color: #111;
    font-family: 'Merriweather', serif;
    margin: 14px 0 5px; padding-bottom: 4px; border-bottom: 1px solid #ebebeb;
}
.bot-bubble h3:first-child { margin-top: 0; }
hr.sep { border: none; border-top: 1px solid #f0f0f0; margin: 8px 0; }
@keyframes rise { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:translateY(0)} }

/* ── TOGGLE BUTTONS ─────────────────────────────────────────────── */
[data-testid="stMain"] [data-testid="stButton"] > button {
    background: #f2f2f2 !important; color: #555 !important;
    -webkit-text-fill-color: #555 !important;
    border: 1px solid #e0e0e0 !important; border-radius: 100px !important;
    padding: 4px 13px !important; font-size: 11.5px !important;
    font-weight: 500 !important; white-space: nowrap !important;
    min-height: unset !important; height: auto !important;
    line-height: 1.5 !important; transition: all .15s !important; box-shadow: none !important;
}
[data-testid="stMain"] [data-testid="stButton"] > button:hover {
    background: #111 !important; color: #fff !important;
    -webkit-text-fill-color: #fff !important; border-color: #111 !important;
}

/* ── PANELS ─────────────────────────────────────────────────────── */
.panel {
    margin: 4px 0 12px 40px; border: 1px solid #e0e0e0; border-radius: 10px;
    overflow: hidden; background: #fff;
    box-shadow: 0 1px 5px rgba(0,0,0,.04); animation: rise .18s ease;
}
.panel-header {
    background: #111; color: #fff; padding: 7px 14px;
    font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing: .08em;
}
.panel-body { padding: 14px 18px; background: #fff; color: #111; font-size: 13px; line-height: 1.75; }
.panel-body * { color: #111 !important; background: transparent !important; }
.panel-body h3 {
    font-size: 13px; font-weight: 700; color: #111 !important;
    font-family: 'Merriweather', serif;
    margin: 16px 0 5px; padding-bottom: 4px; border-bottom: 1px solid #ebebeb;
}
.panel-body h3:first-child { margin-top: 0; }
.panel-body ul, .panel-body ol { padding-left: 18px; }
.panel-body li { color: #111 !important; margin-bottom: 4px; }
.source {
    display: inline-flex; align-items: center; gap: 5px;
    background: #f4f4f4; color: #111 !important; border: 1px solid #e0e0e0;
    padding: 4px 10px; border-radius: 6px; font-size: 11.5px; font-weight: 500;
    margin: 3px 4px 3px 0; max-width: 210px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.source:hover { background: #e8e8e8; }
.kg-legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 11px; color: #666; padding-bottom: 8px; }
.kg-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 3px; vertical-align: middle; }

/* ── FIXED INPUT BAR ────────────────────────────────────────────── */
.input-bar {
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 200;
    pointer-events: none;
    background: linear-gradient(to top, #fff 72%, transparent);
    padding: 4px 0 12px;
}
.input-wrap { max-width: 780px; margin: 0 auto; padding: 0 20px; pointer-events: all; }
.input-chips { display: flex; gap: 6px; flex-wrap: wrap; justify-content: center; margin-top: 8px; }
.chip {
    font-size: 11px; color: #999; background: #f8f8f8;
    border: 1px solid #ebebeb; border-radius: 100px; padding: 3px 11px; white-space: nowrap;
}
.input-note { text-align: center; font-size: 10px; color: #ccc; margin-top: 5px; }

[data-testid="stTextInput"] input,
[data-testid="stTextInput"] input:focus,
[data-testid="stTextInput"] input:hover {
    background: #fff !important; color: #111 !important;
    -webkit-text-fill-color: #111 !important; caret-color: #111 !important;
    border: none !important; outline: none !important; box-shadow: none !important;
    font-family: 'Inter', sans-serif !important; font-size: 14px !important; padding: 6px 0 !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #aaa !important; -webkit-text-fill-color: #aaa !important; opacity: 1 !important;
}
[data-testid="stTextInput"] > div,
[data-testid="stTextInput"] > div > div,
[data-testid="stTextInput"] > div > div > div {
    background: #fff !important; border: none !important; box-shadow: none !important; padding: 0 !important;
}
[data-testid="stTextInput"] label { display: none !important; }

[data-testid="stFormSubmitButton"] > button {
    background: #111 !important; color: #fff !important; -webkit-text-fill-color: #fff !important;
    border: none !important; border-radius: 9px !important; padding: 8px 18px !important;
    font-family: 'Inter', sans-serif !important; font-size: 13px !important;
    font-weight: 600 !important; white-space: nowrap !important; min-height: unset !important;
    line-height: 1 !important; cursor: pointer !important;
    transition: background .15s !important; box-shadow: none !important;
}
[data-testid="stFormSubmitButton"] > button:hover { background: #333 !important; }

[data-testid="stForm"] {
    background: #fff !important; border: 1.5px solid #d0d0d0 !important;
    border-radius: 13px !important; padding: 6px 8px 6px 16px !important;
    box-shadow: 0 3px 16px rgba(0,0,0,.09) !important; transition: border-color .2s !important;
}
[data-testid="stForm"]:focus-within { border-color: #111 !important; }
[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
    display: flex !important; align-items: center !important;
    gap: 8px !important; flex-wrap: nowrap !important; width: 100% !important;
}
[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:first-child {
    flex: 1 1 auto !important; min-width: 0 !important;
}
[data-testid="stForm"] [data-testid="stHorizontalBlock"] > div:last-child { flex: 0 0 auto !important; }
[data-testid="InputInstructions"], [data-testid="stForm"] small,
[data-testid="stProgressBar"] { display: none !important; }

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { color: #111 !important; }

[data-testid="stSelectbox"] label { font-size: 12px !important; color: #666 !important; }
[data-testid="stAlert"] {
    background: #fafafa !important; border: 1px solid #e8e8e8 !important;
    border-radius: 7px !important; color: #666 !important; font-size: 12.5px !important;
}

@media (max-width: 860px) { .panel { margin-left: 0 !important; } .badge { display: none; } }
@media (max-width: 600px) { .user-bubble { max-width: 88%; } .cards-grid { grid-template-columns: 1fr !important; } }
</style>
""", unsafe_allow_html=True)


# SYNC TOGGLES

def sync_toggles():
    """
    Ensures that for every chat message, there are boolean flags to track 
    whether its Sources, Summary, or Knowledge Graph panels are currently expanded.
    """
    for k in ["show_sources", "show_summary", "show_kg"]:
        while len(st.session_state[k]) < len(st.session_state.messages):
            st.session_state[k].append(False)

sync_toggles()


# SIDEBAR

# The sidebar contains the main navigation menu, active session details, 
# and the logout button.

def _nav(label: str, page_key: str):
    """Helper function to render a navigation button that changes the current page."""
    safe = label.replace(" ", "_").replace(".", "").replace("/", "").replace("&", "")
    if st.button(label, key=f"nav_{safe}", use_container_width=True):
        st.session_state.current_page = page_key
        st.rerun()


with st.sidebar:
    st.markdown("### ⚖ LAWPAK")
    st.caption("Pakistan Legal Research Assistant")

    _u = st.session_state.current_user or ""
    _initial = _u[0].upper() if _u else "?"
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:9px;padding:8px 14px;margin:4px 0;
            background:#1e1e1e;border-radius:9px;border:1px solid #2a2a2a;">
  <div style="width:28px;height:28px;background:#333;border-radius:50%;display:flex;
              align-items:center;justify-content:center;font-size:13px;font-weight:700;
              color:#fff;flex-shrink:0;">{_initial}</div>
  <div>
    <div style="font-size:12.5px;font-weight:600;color:#e8e8e8;white-space:nowrap;
                overflow:hidden;text-overflow:ellipsis;">{html_lib.escape(_u)}</div>
    <div style="font-size:10px;color:#555;">Active session</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("**MENU**")
    _nav("🏠  Home", "chat")
    if st.button("＋  New Chat", key="new_chat", use_container_width=True):
        for k in ["messages", "show_sources", "show_summary", "show_kg"]:
            st.session_state[k] = []
        st.session_state.current_page = "chat"
        st.rerun()

    st.markdown("**💬 Recent**")
    st.markdown("Crl.A.1 — Murder PPC 302")
    st.markdown("Bail in terrorism cases")
    st.markdown("Section 302/324 analysis")
    st.markdown("W.P. 1234/2022 petition")

    st.divider()

    st.markdown("**🛠 Core Tools**")
    _nav("🔍  Search Cases", "search")
    _nav("⚖️  Legal Q&A", "chat")
    _nav("📄  Case Summarizer", "summarizer")

    st.divider()

    st.markdown("**🕸️ Knowledge Graph**")
    _nav("🌐  Graph Explorer", "kg_explorer")
    _nav("🔗  Case Relationships", "kg_explorer")

    st.divider()

    st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
    if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.caption("LAWPAK v1.0 · LLaMA + Neo4j")


# TOP BAR

_page_titles = {
    "chat":        ("Legal Research Session",      "Pakistan Judgment Database"),
    "search":      ("Case Search",                 "Search across Pakistan court judgments"),
    "summarizer":  ("Case Summarizer",             "AI-powered judgment summaries"),
    "kg_explorer": ("Knowledge Graph Explorer",    "Visualize case relationships"),
}
_pt, _ps = _page_titles.get(st.session_state.current_page, ("LAWPAK", ""))

st.markdown(f"""
<div class="topbar">
  <div>
    <div class="topbar-title">{_pt}</div>
    <div class="topbar-sub">{_ps}</div>
  </div>
  <div class="badge"><span class="badge-dot"></span>LLaMA · Neo4j · RAG</div>
</div>""", unsafe_allow_html=True)

# 
# PAGE ROUTING
# 
# Render the main content area based on the navigation state selected in the sidebar.

if st.session_state.current_page == "chat":
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

    # ── WELCOME SCREEN ──
    if not st.session_state.messages:
        _uname = html_lib.escape(st.session_state.current_user or "Counselor").title()

        # Hero heading — simple HTML, no nested grid
        st.markdown(f"""
<div class="welcome-hero">
  <div class="welcome-icon">⚖️</div>
  <h1>Hi {_uname},<br>How Can I Assist You Today?</h1>
  <p>Ask about any Pakistan court judgment, case reference,
     legal principle, or statutory provision.</p>
</div>""", unsafe_allow_html=True)

        # Feature cards — use st.columns so Streamlit renders them natively
        _cards = [
            ("📋", "Case Summary",     "Summarize any judgment with key facts, arguments, and decisions extracted"),
            ("🔍", "Legal Query",      "Search PPC sections, bail precedents, or any statute or legal principle"),
            ("🕸️", "Knowledge Graph", "Visualize relationships between cases, judges, sections, and parties"),
            ("⚖️", "Precedent Finder", "Find similar rulings and binding case law relevant to your query"),
        ]
        c1, c2 = st.columns(2, gap="small")
        for idx, (icon, title, desc) in enumerate(_cards):
            col = c1 if idx % 2 == 0 else c2
            with col:
                st.markdown(f"""
<div class="feat-card">
  <div class="feat-icon">{icon}</div>
  <div class="feat-title">{title}</div>
  <div class="feat-desc">{desc}</div>
</div>""", unsafe_allow_html=True)

        # Small spacer before input area
        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    # ── MESSAGES ──
    for i, msg in enumerate(st.session_state.messages):

        if msg["role"] == "user":
            st.markdown(f"""
<div class="user-msg">
  <div class="user-bubble">{html_lib.escape(msg['text'])}</div>
</div>""", unsafe_allow_html=True)
            continue

        st.markdown(f"""
<div class="bot-row">
  <div class="bot-av">⚖</div>
  <div class="bot-bubble">{msg['text']}</div>
</div>""", unsafe_allow_html=True)

        bc1, bc2, bc3, _ = st.columns([1, 1, 1.5, 5.5])
        with bc1:
            if st.button(
                "▾ Sources" if st.session_state.show_sources[i] else "▸ Sources",
                key=f"src_{i}",
            ):
                st.session_state.show_sources[i] = not st.session_state.show_sources[i]
                st.rerun()
        with bc2:
            if st.button(
                "▾ Summary" if st.session_state.show_summary[i] else "▸ Summary",
                key=f"sum_{i}",
            ):
                st.session_state.show_summary[i] = not st.session_state.show_summary[i]
                st.rerun()
        with bc3:
            if st.button(
                "▾ Knowledge Graph" if st.session_state.show_kg[i] else "▸ Knowledge Graph",
                key=f"kg_{i}",
            ):
                st.session_state.show_kg[i] = not st.session_state.show_kg[i]
                st.rerun()

        # Sources panel
        if st.session_state.show_sources[i]:
            st.markdown(
                '<div class="panel"><div class="panel-header">📎 Source Citations</div>'
                '<div class="panel-body">',
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                chips = "".join(
                    f'<span class="source">📄 {html_lib.escape(str(s))}</span>'
                    for s in msg["sources"]
                )
                st.markdown(f'<div style="display:flex;flex-wrap:wrap;">{chips}</div>', unsafe_allow_html=True)
            else:
                st.info("No sources available.")
            st.markdown("</div></div>", unsafe_allow_html=True)

        # Summary panel
        if st.session_state.show_summary[i]:
            st.markdown(
                '<div class="panel"><div class="panel-header">📝 Case Summary</div>'
                '<div class="panel-body">',
                unsafe_allow_html=True,
            )
            if msg.get("summary"):
                txt = msg["summary"]
                for sec in [
                    "Case Background:", "Parties Involved:", "Key Facts:", "Issues:",
                    "Arguments of Both Sides:", "Reasoning of the Court:", "Final Decision:",
                    "Citations:", "Legal Principles Applied:",
                ]:
                    txt = txt.replace(sec, f"\n\n### {sec}\n")
                txt = re.sub(r"\* ", "\n- ", txt)
                txt = re.sub(r"\n{3,}", "\n\n", txt)
                st.markdown(txt)
            else:
                st.info("No summary available.")
            st.markdown("</div></div>", unsafe_allow_html=True)

        # Knowledge Graph panel
        if st.session_state.show_kg[i]:
            st.markdown(
                '<div class="panel"><div class="panel-header">🕸️ Knowledge Graph</div>'
                '<div class="panel-body">',
                unsafe_allow_html=True,
            )
            if msg.get("kg"):
                cases = list(dict.fromkeys(e["from"] for e in msg["kg"]))
                sel = st.selectbox("Select case:", cases, key=f"kg_sel_{i}")
                edges = [e for e in msg["kg"] if e["from"] == sel]
                cmap = {
                    "Judge": "#22c55e", "Section": "#f59e0b",
                    "Decision": "#ef4444", "Petitioner": "#a855f7", "Respondent": "#3b82f6",
                }
                net = Network(height="420px", width="100%", directed=True, bgcolor="#ffffff")
                net.barnes_hut(
                    gravity=-5000, central_gravity=0.5,
                    spring_length=200, spring_strength=0.05, damping=0.09,
                )
                net.add_node(
                    sel, label=sel, color="#111", shape="box", size=30,
                    font={"size": 13, "bold": True, "color": "#fff"},
                )
                for e in edges:
                    c = cmap.get(e.get("type", "Other"), "#888")
                    net.add_node(e["to"], label=e["to"], color=c, size=20, font={"size": 11})
                    net.add_edge(sel, e["to"], label=e["label"].replace("_", " "), font={"size": 9}, arrows="to")
                st.markdown("""
<div class="kg-legend">
  <span><span class="kg-dot" style="background:#22c55e"></span>Judge</span>
  <span><span class="kg-dot" style="background:#f59e0b"></span>Section</span>
  <span><span class="kg-dot" style="background:#ef4444"></span>Decision</span>
  <span><span class="kg-dot" style="background:#a855f7"></span>Petitioner</span>
  <span><span class="kg-dot" style="background:#3b82f6"></span>Respondent</span>
</div>""", unsafe_allow_html=True)
                net.save_graph("kg.html")
                with open("kg.html", "r", encoding="utf-8") as f:
                    components.html(f.read(), height=440)
            else:
                st.info("No knowledge graph data available.")
            st.markdown("</div></div>", unsafe_allow_html=True)

        if i < len(st.session_state.messages) - 1:
            st.markdown('<hr class="sep">', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close .content-wrap

    # ── FIXED INPUT BAR ──
    st.markdown('<div class="input-bar"><div class="input-wrap">', unsafe_allow_html=True)

    with st.form("chat", clear_on_submit=True):
        c1, c2 = st.columns([10, 1])
        with c1:
            query = st.text_input("", placeholder="Ask about a case, statute, or legal principle…")
        with c2:
            send = st.form_submit_button("Send ➤")

    st.markdown("""
<div class="input-chips">
  <span class="chip">⚖️ Legal Q&amp;A</span>
  <span class="chip">📋 Summarize</span>
  <span class="chip">🔍 Find Cases</span>
  <span class="chip">🕸️ Deep Research</span>
</div>
<div class="input-note">LAWPAK may make mistakes — verify with official court records.</div>
</div></div>""", unsafe_allow_html=True)

    # ── HANDLE SUBMIT ──
    if send and query.strip():
        q = query.strip()
        st.session_state.messages.append({"role": "user", "text": q})
        for k in ["show_sources", "show_summary", "show_kg"]:
            st.session_state[k].append(False)

        with st.spinner("Researching…"):
            result = ask_question(q)

        answer = result.get("answer", "No answer returned.")

        slot = st.empty()
        buf = ""
        for word in answer.split():
            buf += word + " "
            slot.markdown(f"""
<div class="bot-row">
  <div class="bot-av">⚖</div>
  <div class="bot-bubble">{buf}▌</div>
</div>""", unsafe_allow_html=True)
            time.sleep(0.012)
        slot.empty()

        st.session_state.messages.append({
            "role": "bot",
            "text": answer,
            "sources": result.get("sources", []),
            "summary": result.get("summary"),
            "kg": result.get("kg"),
        })
        for k in ["show_sources", "show_summary", "show_kg"]:
            st.session_state[k].append(False)

        st.session_state["_scroll"] = True
        st.rerun()

    if st.session_state.get("_scroll"):
        st.session_state["_scroll"] = False
        components.html("""
<script>
  const m = window.parent.document.querySelector('[data-testid="stMain"]');
  if (m) m.scrollTo({ top: m.scrollHeight, behavior: 'smooth' });
  window.parent.scrollTo({ top: window.parent.document.body.scrollHeight, behavior: 'smooth' });
</script>""", height=0)


# NON-CHAT PAGES

elif st.session_state.current_page == "search":
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    st.markdown("""
<div class="welcome-hero">
  <div class="welcome-icon">🔍</div>
  <h1>Case Search</h1>
  <p>Search across Pakistan court judgments by section, keyword, or case number.</p>
</div>""", unsafe_allow_html=True)
    search_q = st.text_input("Search query", placeholder="e.g. Section 302 PPC, murder appeal…")
    if search_q:
        st.info("Search is connected to the same RAG pipeline. Try asking in the chat for now.")
        if st.button("→ Ask in Chat", key="search_to_chat"):
            st.session_state.current_page = "chat"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.current_page == "summarizer":
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    st.markdown("""
<div class="welcome-hero">
  <div class="welcome-icon">📄</div>
  <h1>Case Summarizer</h1>
  <p>Get AI-generated summaries of court judgments with key facts, arguments, and decisions extracted.</p>
</div>""", unsafe_allow_html=True)
    case_ref = st.text_input("Case reference", placeholder="e.g. Crl.A.1, W.P. 1234/2022…")
    if case_ref:
        st.info("Summaries are generated via the chat. Try asking: 'Summarize " + case_ref + "'")
        if st.button("→ Ask in Chat", key="sum_to_chat"):
            st.session_state.current_page = "chat"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.current_page == "kg_explorer":
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    st.markdown("""
<div class="welcome-hero">
  <div class="welcome-icon">🕸️</div>
  <h1>Knowledge Graph Explorer</h1>
  <p>Visualize relationships between cases, judges, sections, and legal principles in an interactive graph.</p>
</div>""", unsafe_allow_html=True)
    kg_path = os.path.join(os.path.dirname(__file__), "kg.html")
    if os.path.exists(kg_path):
        with open(kg_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=600)
    else:
        st.info("No knowledge graph generated yet. Ask a question in Chat to generate one.")
        if st.button("→ Go to Chat", key="kg_to_chat"):
            st.session_state.current_page = "chat"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
