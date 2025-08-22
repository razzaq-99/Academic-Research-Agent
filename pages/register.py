import streamlit as st
import os
import json
import re
import uuid
import hashlib
import binascii
import datetime

USERS_FILE = "users.json"
APP_TITLE = "Smart Research Hub"
APP_SUBTITLE = "Create an account to save and manage your research."

# ----------------- HELPERS (same as login) ----------------

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


def find_user_by_email(email: str):
    users = load_users()
    for u in users:
        if u.get("email", "").lower() == email.lower():
            return u
    return None


# ------------------ Custom CSS for register page ------------------

st.set_page_config(page_title=f"Register | {APP_TITLE}", layout="centered")

CUSTOM_CSS = """
<style>
:root{
  --primary1:#f36b6b;
  --primary2:#b71c1c;
  --card:#ffffff;
  --muted:#6b7280;
  --shadow: rgba(13,17,28,0.10);
}

/* page background and spacing */
[data-testid="stAppViewContainer"]{
    background: linear-gradient(180deg, #fbfcfd, #f4f6fa) !important;
}

.block-container{
    padding-top: 40px; /* same breathing room as login */
    padding-bottom: 28px;
}

/* header styling (matches login) */
.app-header{
    text-align: center;
    margin-top: 18px;
    margin-bottom: 14px;
    padding-left: 12px;
    padding-right: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
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

/* center alignment for the card area */
.login-col{
    display:flex;
    justify-content:center;
    align-items:flex-start;
}

/* card style (keeps spacing consistent with login) */
.card{
    width: 0px;
    max-width: 100vw;
    background: var(--card);
    border-radius: 20px;
    padding: 28px;
    margin-top: 20px;
}

/* center the form heading inside the card */
.card h3{
    text-align:center;
    margin-top: 6px;
    margin-bottom: 18px;
    font-size:18px;
    color: #101428;
}

/* inputs styling consistent with login */
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

.card .stTextInput>div>div>input::placeholder { color:#6b7280 !important; }

/* center the create-account button inside the card */
.card .stButton>button {
    background: linear-gradient(180deg,var(--primary2), #8b0000);
    color: white;
    padding: 10px 32px;
    border-radius: 10px;
    width: auto;
    font-weight: 700;
    border: none;
    box-shadow: 0 6px 18px rgba(183,28,28,0.12);
    display: block;
    margin: 0 auto; /* centers the button */
}

/* keep checkbox styling */
.card .stCheckbox>div label { color:#0b1220 !important; }

/* footer / small text */
.small-muted{
    color: var(--muted);
    font-size:13px;
    text-align:center;
    margin-top:12px;
}

/* link styling (same class name as login for consistency) */
a.register-link {
    color: var(--primary2);
    font-weight:600;
    text-decoration: none;
    padding: 8px 14px;
    border-radius: 10px;
    border: 1px solid rgba(183,28,28,0.06);
}
a.register-link:hover { text-decoration: underline; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------ Header ------------------
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


left_col, center_col, right_col = st.columns([1, 3, 0.8])

with center_col:
    st.markdown("<div class='login-col'>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # ------------------ Registration form ------------------
    with st.form("register_form"):
        st.markdown("<h3>Create an account</h3>", unsafe_allow_html=True)
        name = st.text_input("Full name")
        email = st.text_input("Email", placeholder="you@university.edu")
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm password", type="password")
        affiliation = st.text_input("Affiliation (optional)")
        accept = st.checkbox("I agree to the Terms & Privacy")

        submitted = st.form_submit_button("🔐 Create account")

        if submitted:
            # ------------------ Form Validation ------------------
            if not name or not email or not password or not password2:
                st.error("Please complete all required fields.")
            elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.error("Please enter a valid email address.")
            elif password != password2:
                st.error("Passwords do not match.")
            elif len(password) < 8:
                st.error("Password should be at least 8 characters long.")
            elif not accept:
                st.error("You must accept Terms & Privacy to create an account.")
            elif find_user_by_email(email):
                st.error("An account with this email already exists. Try logging in.")
            else:
                users = load_users()
                uid = uuid.uuid4().hex
                salt, pwd_hash = hash_password(password)
                user = {
                    "id": uid,
                    "name": name.strip(),
                    "email": email.strip().lower(),
                    "affiliation": affiliation.strip(),
                    "salt": salt,
                    "password_hash": pwd_hash,
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                }
                users.append(user)
                save_users(users)
                st.success("Account created successfully! You can now go to the Login page and sign in.")
                st.info("Tip: For production, replace local storage with Supabase/Auth0 and implement email verification + password reset flows.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("\n")
st.caption("This registration stores user data locally in `users.json` for prototyping. Do not use this method for public deployments.")


st.markdown(
    "<p class='small-muted'>Already have an account? <a class='register-link' href='?page=Login' target='_self'>Login here</a></p>",
    unsafe_allow_html=True,
)


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

# ----- End of register.py -----
