import streamlit as st
import os
import json
import re
import uuid
import hashlib
import binascii
import datetime

# ---------------- CONFIG ----------------
USERS_FILE = "users.json"
APP_TITLE = "Smart Research Hub"
APP_SUBTITLE = "Your AI-Powered Research Companion"

# ----------------- HELPERS ----------------

def _ensure_users_file():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({"users": []}, f)


def load_users():
    _ensure_users_file()
    with open(USERS_FILE, "r") as f:
        data = json.load(f)
    return data.get("users", [])


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump({"users": users}, f, indent=2)


def hash_password(password: str, salt: str = None):
    
    if salt is None:
        salt = os.urandom(16).hex()
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200000)
    return salt, binascii.hexlify(dk).decode()


def verify_password(salt: str, pwd_hash: str, password: str):
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200000)
    return binascii.hexlify(dk).decode() == pwd_hash


def find_user_by_email(email: str):
    users = load_users()
    for u in users:
        if u.get("email", "").lower() == email.lower():
            return u
    return None


def authenticate(email: str, password: str):
    user = find_user_by_email(email)
    if not user:
        return False, "No account found with this email."
    if verify_password(user.get("salt"), user.get("password_hash"), password):
        return True, user
    return False, "Incorrect password."


# ----------------- UI / STUFF ----------------

# Use centered layout so Streamlit won't add odd global spacing
st.set_page_config(page_title=f"Login | {APP_TITLE}", layout="centered")

CUSTOM_CSS = """
<style>
:root{
  --primary1:#f36b6b;
  --primary2:#b71c1c;
  --card:#ffffff;
  --muted:#6b7280;
  --shadow: rgba(13,17,28,0.10);
}

/* App background (keeps light gradient so card stands out regardless of theme) */
[data-testid="stAppViewContainer"]{
    background: linear-gradient(180deg, #fbfcfd, #f4f6fa) !important;
}

/* Block container spacing */
.block-container{
    padding-top: 18px;
    padding-bottom: 28px;
}

/* Header (centered, modern) */
.app-header{
    text-align: center;
    margin-bottom: 6px;
    padding-left: 12px;
    padding-right: 12px;
}
.header-icon{
    font-size:42px;
    line-height:1;
    margin-bottom:6px;
}
.header-title{
    font-size:26px;
    font-weight:800;
    margin:0;
    display:inline-block;
    background: linear-gradient(90deg,var(--primary1),var(--primary2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-sub{
    font-size:12px;
    color:var(--muted);
    margin-top:6px;
}

/* Layout: use columns to center the card reliably */
.login-col{
    display:flex;
    justify-content:center;
    align-items:flex-start;
}

/* Card container */
# .card{
#     width: 0px;
#     max-width: 100vw;
#     background: var(--card);
#     border-radius: 14px;
#     padding: 28px;
#     box-shadow: 0 10px 28px var(--shadow);
#     border: 1px solid rgba(16,24,40,0.03);
#     margin-top: 20px;
# }
.card h3{
    text-align:left;
    margin-bottom: 18px;
    font-size:18px;
    color: #101428;
}

/* Only style inputs that are inside the card so other components remain untouched */
.card .stTextInput>div>div>input,
.card .stTextInput>div>div>textarea {
    border-radius:10px !important;
    padding:12px !important;
    height:44px;
    border:1px solid rgba(16,24,40,0.06) !important;
    background:#ffffff !important;
    color:#0b1220 !important;
}
.card .stTextInput>label { color: #0b1220 !important; font-weight:600; }

/* placeholder */
.card .stTextInput>div>div>input::placeholder { color:#6b7280 !important; }

/* Button: scoped styling for card buttons */
.card .stButton>button {
    background: linear-gradient(180deg,var(--primary2), #8b0000);
    color: white;
    padding: 10px 14px;
    border-radius: 10px;
    width: 100%;
    font-weight: 700;
    border: none;
    box-shadow: 0 6px 18px rgba(183,28,28,0.12);
}

/* checkbox label color inside card */
.card .stCheckbox>div label { color:#0b1220 !important; }

/* Muted footer */
.small-muted{
    color: var(--muted);
    font-size:13px;
    text-align:center;
    margin-top:12px;
}

/* Link style */
a.register-link {
    color: var(--primary2);
    font-weight:600;
    text-decoration: none;
}
a.register-link:hover { text-decoration: underline; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------- Sidebar ----------------

# (Sidebar removed for a focused login page)

# ----------------- Header ----------------
st.markdown(
    f"""
    <div class='app-header'>
        <div class='header-icon'></div>
        <div class='header-title'>{APP_TITLE}</div>
        <div class='header-sub'>{APP_SUBTITLE}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Use columns to reliably center the card across environments (left & right columns are flexible)
left_col, center_col, right_col = st.columns([1, 3, 1])

with center_col:
    st.markdown("<div class='login-col'>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Login to Continue")

    with st.form("login_form"):
        email = st.text_input("Email", placeholder="you@university.edu")
        password = st.text_input("Password", type="password")
        remember = st.checkbox("Remember me")
        submitted = st.form_submit_button("🔓 Sign in")

        if submitted:
            if not email or not password:
                st.error("Please provide both email and password.")
            else:
                ok, payload = authenticate(email.strip(), password)
                if ok:
                    st.session_state["user"] = {
                        "id": payload.get("id"),
                        "name": payload.get("name"),
                        "email": payload.get("email"),
                    }
                    st.success(f"Welcome back, {payload.get('name') or payload.get('email')}!")
                    st.write("You are now authenticated in the Streamlit session.\n")
                    st.info("Next: wire this session to your main app. Typically: st.session_state['user']['id'] is used as the per-user namespace for vectorstore retrieval.")
                else:
                    st.error(payload)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# small footer link centered on full width
st.markdown("<p class='small-muted'>Don't have an account? <a class='register-link' href='/Register' target='_self'>Register here</a></p>", unsafe_allow_html=True)

# Small defensive script: remove any unusually large input element that is NOT inside our .card
# (this targets the mysterious white bar that can appear because of stray inputs from other pages/themes;
#  it only hides inputs wider than 500px that are not descendants of .card)
st.markdown(
    """
    <script>
    window.addEventListener('load', function(){
      setTimeout(function(){
        try {
          document.querySelectorAll('input').forEach(function(inp){
            var rect = inp.getBoundingClientRect();
            // if an input is very wide and not inside .card, hide it (defensive - fixes stray white bar)
            if(rect.width > 500 && !inp.closest('.card')) {
              inp.style.display = 'none';
            }
          });
        } catch(e) { /* fail silently */ }
      }, 160);
    });
    </script>
    """,
    unsafe_allow_html=True,
)
