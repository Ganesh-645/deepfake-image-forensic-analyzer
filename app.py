import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sqlite3
import hashlib
import os
import time
import datetime

# ─────────────────────────────────────────────
# DATABASE SETUP  (SQLite – persists across restarts)
# ─────────────────────────────────────────────
DB_PATH = "users.db"
MAX_FILE_SIZE_MB = 10
SESSION_TIMEOUT_MINUTES = 60
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 300   # 5 minutes


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT UNIQUE NOT NULL,
                password  TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS login_attempts (
                username    TEXT NOT NULL,
                attempt_at  REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                image_name TEXT,
                ela_score REAL,
                fft_score REAL,
                combined_score REAL,
                prediction TEXT,
                confidence REAL,
                analyzed_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


init_db()

# ─────────────────────────────────────────────
# PASSWORD HELPERS  (SHA-256 + salt – no extra deps)
# ─────────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, hashed = stored.split(":")
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed
    except Exception:
        return False


# ─────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────
def get_recent_attempts(username: str) -> int:
    cutoff = time.time() - LOCKOUT_SECONDS
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM login_attempts WHERE username=? AND attempt_at>?",
            (username, cutoff)
        ).fetchone()
    return row[0]


def record_failed_attempt(username: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO login_attempts (username, attempt_at) VALUES (?,?)",
            (username, time.time())
        )
        conn.commit()


def clear_attempts(username: str):
    with get_db() as conn:
        conn.execute("DELETE FROM login_attempts WHERE username=?", (username,))
        conn.commit()


def create_user(username: str, password: str) -> tuple:
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?,?)",
                (username.strip().lower(), hash_password(password))
            )
            conn.commit()
        return True, "Account created! You can now log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists. Choose another."


def authenticate(username: str, password: str) -> tuple:
    username = username.strip().lower()
    attempts = get_recent_attempts(username)
    if attempts >= MAX_LOGIN_ATTEMPTS:
        remaining = int(LOCKOUT_SECONDS / 60)
        return False, f"Too many failed attempts. Account locked for {remaining} minutes."
    with get_db() as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username=?", (username,)
        ).fetchone()
    if row and verify_password(password, row["password"]):
        clear_attempts(username)
        return True, "ok"
    record_failed_attempt(username)
    left = MAX_LOGIN_ATTEMPTS - attempts - 1
    return False, f"Invalid username or password. {left} attempt(s) remaining before lockout."

def save_analysis(
    username,
    image_name,
    ela,
    fft,
    combined,
    prediction,
    confidence
):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO analysis_history
            (
                username,
                image_name,
                ela_score,
                fft_score,
                combined_score,
                prediction,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            username,
            image_name,
            ela,
            fft,
            combined,
            prediction,
            confidence
        ))
        conn.commit()


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for key, default in [
    ("logged_in", False),
    ("username", ""),
    ("login_time", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def is_session_expired() -> bool:
    if st.session_state.login_time is None:
        return False
    elapsed = (datetime.datetime.now() - st.session_state.login_time).total_seconds()
    return elapsed > SESSION_TIMEOUT_MINUTES * 60


if st.session_state.logged_in and is_session_expired():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.login_time = None
    st.warning("Your session has expired. Please log in again.")


# ─────────────────────────────────────────────
# AUTH UI
# ─────────────────────────────────────────────
def login_form():
    st.subheader("Log in to your account")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Log in")
        if login_btn:
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                ok, msg = authenticate(username, password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip().lower()
                    st.session_state.login_time = datetime.datetime.now()
                    st.rerun()
                else:
                    st.error(msg)


def signup_form():
    st.subheader("Create an account")
    with st.form("signup_form", clear_on_submit=True):
        new_user = st.text_input("Username (min 3 characters)")
        new_pass = st.text_input("Password (min 6 characters)", type="password")
        confirm  = st.text_input("Confirm password", type="password")
        signup_btn = st.form_submit_button("Sign up")
        if signup_btn:
            if new_pass != confirm:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_user(new_user, new_pass)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


def auth_ui():
    st.title("Deepfake Analyzer Portal")
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])
    with tab_login:
        login_form()
    with tab_signup:
        signup_form()


def logout_button():
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
    if st.session_state.login_time:
        elapsed = (datetime.datetime.now() - st.session_state.login_time)
        remaining = SESSION_TIMEOUT_MINUTES - int(elapsed.total_seconds() / 60)
        st.sidebar.caption(f"Session expires in ~{max(remaining,0)} min")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.login_time = None
        st.rerun()


if not st.session_state.logged_in:
    auth_ui()
    st.stop()

logout_button()

page = st.sidebar.radio(
    "Navigation",
    [
        "Analyzer",
        "Detection History"
    ]
)

# ─────────────────────────────────────────────
# FILE VALIDATION
# ─────────────────────────────────────────────
def validate_file(uploaded) -> tuple:
    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB."
    ext = uploaded.name.lower().split(".")[-1]
    if ext not in {"jpg", "jpeg", "png"}:
        return False, f"Unsupported format: .{ext}. Only JPG and PNG accepted."
    return True, "ok"


# ─────────────────────────────────────────────
# ELA  (PNG-safe)
# ─────────────────────────────────────────────
@st.cache_data
def compute_ela(image: Image.Image, quality: int = 90):
    rgb = image.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")
    ela = ImageChops.difference(rgb, compressed)
    extrema = ela.getextrema()
    max_diff = max(ex[1] for ex in extrema) if extrema else 0
    scale = 255.0 / max_diff if max_diff != 0 else 1.0
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    return np.array(ela)[..., 0]


# ─────────────────────────────────────────────
# FFT
# ─────────────────────────────────────────────
@st.cache_data
def compute_fft(image: Image.Image):
    gray = np.array(image.convert("L"))
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    return 20 * np.log(np.abs(fshift) + 1)


# ─────────────────────────────────────────────
# SCORING + CLASSIFICATION
# ─────────────────────────────────────────────
def ela_score(ela_arr) -> float:
    return float(np.mean(ela_arr) / 255.0 * 80)


def fft_score(spectrum) -> float:
    threshold = np.mean(spectrum) + np.std(spectrum)
    return float(np.sum(spectrum > threshold) / spectrum.size * 100)


def classify(combined: float) -> tuple:
    if combined <= 10:
        return "Authentic Image", min(95 - combined, 99), "success"
    elif combined <= 15:
        return "AI Manipulated Image", min(75 + combined, 99), "warning"
    else:
        return "Fully AI Generated Image", min(90 + combined / 5, 99), "error"


# ─────────────────────────────────────────────
# REPORT DOWNLOAD
# ─────────────────────────────────────────────
def build_report(results: list) -> bytes:
    lines = [
        "DEEPFAKE IMAGE FORENSIC ANALYZER – ANALYSIS REPORT",
        f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Analyst   : {st.session_state.username}",
        "=" * 60,
    ]
    for r in results:
        lines += [
            "",
            f"File       : {r['name']}",
            f"ELA Score  : {r['ela']:.2f}",
            f"FFT Score  : {r['fft']:.2f}",
            f"Combined   : {r['combined']:.2f}",
            f"Prediction : {r['label']}",
            f"Confidence : {r['conf']:.1f}%",
        ]
    lines += [
        "",
        "=" * 60,
        "DISCLAIMER: Results are forensic indicators, not definitive proof.",
        "Accuracy may vary for heavily compressed or post-processed images.",
    ]
    return "\n".join(lines).encode("utf-8")

def show_history():

    st.title("Detection History")

    with get_db() as conn:
        rows = conn.execute("""
            SELECT *
            FROM analysis_history
            WHERE username = ?
            ORDER BY analyzed_at DESC
        """, (
            st.session_state.username,
        )).fetchall()

    if not rows:
        st.info("No previous detections found.")
        return

    for row in rows:

        st.markdown("---")

        st.write(f"Date: {row['analyzed_at']}")
        st.write(f"Image: {row['image_name']}")
        st.write(f"Prediction: {row['prediction']}")
        st.write(f"Confidence: {row['confidence']:.1f}%")
        st.write(f"ELA Score: {row['ela_score']:.2f}")
        st.write(f"FFT Score: {row['fft_score']:.2f}")

# ─────────────────────────────────────────────
# MAIN ANALYSIS UI
# ─────────────────────────────────────────────
if page == "Detection History":
    show_history()
    st.stop()

st.title("Deepfake Image Forensic Analyzer")
st.write(
    "Upload one or more images (JPG/PNG, max 10 MB each) to analyze "
    "compression artifacts and frequency patterns for deepfake detection."
)

uploaded_files = st.file_uploader(
    "Upload JPG / PNG Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    results = []

    for idx, uploaded in enumerate(uploaded_files, start=1):
        st.markdown("---")
        st.subheader(f"Image {idx}: {uploaded.name}")

        valid, msg = validate_file(uploaded)
        if not valid:
            st.error(f"Skipped — {msg}")
            continue

        image = Image.open(uploaded).convert("RGB")
        is_png = uploaded.name.lower().endswith(".png")

        st.image(image, caption="Uploaded Image", width=500)

        if is_png:
            st.info(
                "PNG detected: converted to JPEG internally for ELA. "
                "ELA results on PNGs are less reliable — treat with caution."
            )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Error Level Analysis (ELA)")
            ela_arr = compute_ela(image)
            st.image(ela_arr, caption="Bright areas = compression anomalies", width=300)

        with col2:
            st.subheader("Frequency Spectrum (FFT)")
            spectrum = compute_fft(image)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(spectrum, cmap="inferno")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        ela_s    = ela_score(ela_arr)
        fft_s    = fft_score(spectrum)
        combined = ela_s * 0.4 + fft_s * 0.6
        label, conf, level = classify(combined)

        if level == "success":
            st.success(f"Prediction: {label}")
        elif level == "warning":
            st.warning(f"Prediction: {label}")
        else:
            st.error(f"Prediction: {label}")

        st.subheader("Detection Details")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ELA Score",      f"{ela_s:.2f}")
        c2.metric("FFT Score",      f"{fft_s:.2f}")
        c3.metric("Combined Score", f"{combined:.2f}")
        c4.metric("Confidence",     f"{conf:.1f}%")

        save_analysis(
            st.session_state.username,
            uploaded.name,
            ela_s,
            fft_s,
            combined,
            label,
            conf
        )

        results.append({
            "name": uploaded.name,
            "ela": ela_s, "fft": fft_s,
            "combined": combined,
            "label": label, "conf": conf,
        })

    if results:
        st.markdown("---")
        st.download_button(
            label="Download Analysis Report (.txt)",
            data=build_report(results),
            file_name=f"forensic_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

else:
    st.info("Upload one or more images to start analysis.")


# ─────────────────────────────────────────────
# SIDEBAR INFO
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### How it works")
st.sidebar.markdown(
    "**ELA** re-compresses the image and highlights regions where "
    "compression behaves abnormally — a common artifact of editing or AI generation.\n\n"
    "**FFT** converts the image to frequency space. AI-generated images "
    "often contain unusual spectral patterns not present in real photographs."
)
st.sidebar.markdown("---")
st.sidebar.caption("Deepfake Forensic Analyzer v2.0")
