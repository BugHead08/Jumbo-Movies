import streamlit as st
import joblib
import numpy as np
import base64
import os
from pathlib import Path

# ======================
# SET PAGE CONFIG
# ======================
st.set_page_config(page_title="🎬 Jumbo Movie: Sentiment & Cluster Analyzer", layout="wide", page_icon="🎥")

BASE_DIR = Path(__file__).resolve().parent

# ======================
# ADD BACKGROUND IMAGE
# ======================
def add_bg_from_local(image_file: str):
    """Set a custom background image with 68% opacity from a local file."""
    fp = BASE_DIR / image_file
    if not fp.exists():
        return
    with open(fp, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.68), rgba(0,0,0,0.68)), 
                        url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Optional background (place 'jumbo.jpg' alongside this file if you want it)
add_bg_from_local("jumbo.jpg")

# ======================
# LOAD MODELS (cached)
# ======================
@st.cache_resource
def load_vectorizer():
    return joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")

@st.cache_resource
def load_nb():
    return joblib.load(BASE_DIR / "classification_models" / "naive_bayes_model.pkl")

@st.cache_resource
def load_svm():
    return joblib.load(BASE_DIR / "classification_models" / "svm_model.pkl")

@st.cache_resource
def load_lr():
    return joblib.load(BASE_DIR / "classification_models" / "logreg_model.pkl")

def load_local_vectorizer(model_code: str, sentiment_lower: str):
    return joblib.load(BASE_DIR / "cluster_models" / f"tfidf_{model_code}_{sentiment_lower}.pkl")

def load_kmeans(model_code: str, sentiment_lower: str):
    return joblib.load(BASE_DIR / "cluster_models" / f"kmeans_{model_code}_{sentiment_lower}.pkl")

# ======================
# CURATED THEMES (PER-MODEL, PER-SENTIMENT)
# ======================
THEME_MAP = {
    "positive": {
        "nb": [
            {"label": "🌟 Visual & Production", "desc": "Praise for visuals, color grading, animation, and overall production quality."},
            {"label": "💡 Story & Inspiration", "desc": "Positive notes on the plot, moral messages, and inspiring takeaways."}
        ],
        "svm": [
            {"label": "🎥 Cinematic Quality", "desc": "Admiration for cinematography, editing, and technical execution."},
            {"label": "❤️ Heartwarming Story", "desc": "Emphasis on uplifting themes, emotions, and memorable moments."}
        ],
        "lr": [
            {"label": "🎼 Music & Atmosphere", "desc": "Highlights about soundtrack, ambient sound, and the emotional tone it creates."},
            {"label": "👨‍👩‍👧 Character Moments", "desc": "Appreciation of character development, chemistry, and heartfelt scenes."}
        ],
    },
    "negative": {
        "nb": [
            {"label": "🧩 Weak Plot", "desc": "Criticism about a confusing, slow, or unconvincing storyline."},
            {"label": "🔊 Technical Issues", "desc": "Complaints about visuals, sound, pacing, or overall execution."}
        ],
        "svm": [
            {"label": "😐 Underwhelming Experience", "desc": "Viewers felt it was boring, unfunny, or failed to impress."}
            ,
            {"label": "📚 Unoriginal / Flawed Details", "desc": "Comments on clichés, lack of originality, or inconsistent details."}
        ],
        "lr": [
            {"label": "⏳ Pacing & Length", "desc": "Critiques about being too slow, too long, or poorly timed transitions."},
            {"label": "📝 Dialogue & Logic Gaps", "desc": "Notes on awkward lines, inconsistencies, or plot holes that break immersion."}
        ],
    }
}

def get_cluster_info(sentiment_lower: str, model_code: str, cluster_index: int) -> dict:
    """
    Return one of two themes (index 0 or 1) for the given sentiment and model.
    Any cluster index is mapped to 0 or 1 using modulo.
    """
    default = {"label": f"Topic {cluster_index % 2}", "desc": "No description available."}
    sentiment_block = THEME_MAP.get(sentiment_lower)
    if not sentiment_block:
        return default
    model_block = sentiment_block.get(model_code)
    if not model_block or len(model_block) < 2:
        return default
    idx = int(cluster_index) % 2
    return model_block[idx]

# ======================
# STREAMLIT UI
# ======================
st.markdown("""
    <h1 style='text-align: center; font-size: 48px; color:white;'>🎬 Sentiment & Topic Cluster Analysis From Jumbo Movies</h1>
    <p style='text-align: center; font-size: 20px; color:white;'>
        Predict the sentiment and discover the underlying theme of tweets using machine learning.<br>
        Built for academic presentations and professional clarity.
    </p>
""", unsafe_allow_html=True)
st.markdown("---")

# Controls
st.subheader("📝 Enter a Text")
input_text = st.text_input("", placeholder="Type a review or tweet here...")
model_choice = st.radio(
    "🧠 Choose Classification Model:",
    ["Naive Bayes", "Support Vector Machine", "Logistic Regression"],
    horizontal=True
)

if st.button("🚀 Analyze Now"):
    if input_text.strip():
        # Load shared vectorizer
        try:
            tfidf_vectorizer = load_vectorizer()
        except Exception as e:
            st.error("❌ Failed to load the global TF-IDF vectorizer.")
            st.exception(e)
            st.stop()

        # Vectorize once; create dense once (SVM/LR use dense)
        try:
            vec = tfidf_vectorizer.transform([input_text])
        except Exception as e:
            st.error("❌ TF-IDF transform failed.")
            st.exception(e)
            st.stop()

        vec_dense = vec.toarray()

        # ---- pick model + input matrix
        try:
            if model_choice == "Naive Bayes":
                model = load_nb()
                model_code = "nb"
                vec_input = vec                           # NB works with sparse
            elif model_choice == "Support Vector Machine":
                model = load_svm()
                model_code = "svm"
                vec_input = vec_dense                     # SVM (SVC) safe with dense
            else:  # Logistic Regression
                try:
                    model = load_lr()
                except FileNotFoundError:
                    st.error("❌ Logistic Regression model not found. Train it first to enable this option.")
                    st.stop()
                model_code = "lr"
                vec_input = vec_dense                     # LR expects dense
        except FileNotFoundError as e:
            st.error("❌ Selected classifier model was not found. Make sure you've trained and saved it in 'classification_models/'.")
            st.exception(e)
            st.stop()
        except Exception as e:
            st.error("❌ Failed to load the selected classifier model.")
            st.exception(e)
            st.stop()

        # ---- predict sentiment
        try:
            sentiment = model.predict(vec_input)[0]
            sentiment_lower = str(sentiment).lower()
        except Exception as e:
            st.error("❌ Prediction failed.")
            st.exception(e)
            st.stop()

        # ---- probability / confidence
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = float(np.max(model.predict_proba(vec_input)))
            except Exception:
                prob = None
        if prob is None:
            prob = 1.0  # safe fallback

        # ---- classification result card
        palettes = {
            'positive': {'emoji': '😊', 'label': 'POSITIVE', 'bg': '#d1f5d3', 'text': '#155724', 'border': '#28a745'},
            'negative': {'emoji': '😟', 'label': 'NEGATIVE', 'bg': '#fcdcdc', 'text': '#721c24', 'border': '#dc3545'}
        }
        v = palettes.get(sentiment_lower, {'emoji': '❓', 'label': 'UNKNOWN', 'bg': '#eeeeee', 'text': '#333', 'border': '#999'})

        st.markdown("## 📊 Sentiment Result")
        st.markdown(
            "<div style='background-color:{}; padding: 40px; border-left: 12px solid {}; border-radius: 12px; color:{}; text-align:center'>"
            "<div style='font-size: 100px;'>{}</div>"
            "<div style='font-size: 50px; font-weight: bold; margin-top: 10px;'>{}</div>"
            "<div style='font-size: 24px; margin-top: 15px;'>Confidence Score: <code>{:.5f}</code></div>"
            "<div style='font-size: 22px;'>Model Used: <b>{}</b></div>"
            "</div>".format(v["bg"], v["border"], v["text"], v["emoji"], v["label"], prob, model_choice),
            unsafe_allow_html=True
        )

        # ---- Cluster section
        st.markdown("---")
        st.markdown("## 🧠 Cluster Assignment")

        if sentiment_lower not in ("positive", "negative"):
            st.info("Cluster themes are defined only for **positive** and **negative** sentiments.")
        else:
            try:
                # STRICT: use artifacts of the chosen classifier only
                tfidf_local = load_local_vectorizer(model_code, sentiment_lower)
                kmeans = load_kmeans(model_code, sentiment_lower)

                # Predict cluster
                vec_cluster = tfidf_local.transform([input_text]).toarray()
                cluster = int(kmeans.predict(vec_cluster)[0])

                # Map to curated themes for the chosen model
                info = get_cluster_info(sentiment_lower, model_code, cluster)

                st.markdown(
                    "<div style='background-color: #e2e8f0; padding: 35px; border-left: 12px solid #0d6efd;"
                    "border-radius: 10px; color: #1a202c;'>"
                    f"<div style='font-size: 36px; font-weight: bold;'>Cluster #{cluster}</div>"
                    f"<div style='font-size: 30px; font-weight: 600; margin-bottom: 15px;'>Topic: {info['label']}</div>"
                    f"<div style='font-size: 22px;'>{info['desc']}</div>"
                    "</div>",
                    unsafe_allow_html=True
                )

            except FileNotFoundError:
                senti_upper = sentiment_lower.upper()
                st.error("❌ Cluster artifacts for the selected model are missing.")
                st.write(f"Expected files for **{model_choice} / {senti_upper}**:")
                st.code(
                    f"cluster_models/tfidf_{model_code}_{sentiment_lower}.pkl\n"
                    f"cluster_models/kmeans_{model_code}_{sentiment_lower}.pkl"
                )
            except Exception as e:
                st.error("❌ Failed to load cluster model or TF-IDF.")
                st.exception(e)
    else:
        st.warning("⚠️ Please enter a tweet before proceeding.")
