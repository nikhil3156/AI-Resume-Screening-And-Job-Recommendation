import streamlit as st
import re
import pickle
import os
from pdfminer.high_level import extract_text

# ----------------------------
# 1️⃣ Load Models Safely
# ----------------------------
def load_model(file_name):
    """Helper to safely load model files."""
    model_path = os.path.join("models", file_name)
    if not os.path.exists(model_path):
        st.error(f"❌ File not found: {model_path}")
        st.stop()
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Load all required models
rf_classifier_categorization = load_model("rf_pipeline_categorization.pkl")
tfidf_vectorizer_categorization = load_model("tfidf_vectorizer_categorization.pkl")
rf_classifier_job_recommendation = load_model("rf_classifier_job_recommendation.pkl")
tfidf_vectorizer_job_recommendation = load_model("tfidf_vectorizer_job_recommendation.pkl")

# ----------------------------
# 2️⃣ Resume Text Processing
# ----------------------------
def cleanResume(txt):
    txt = re.sub(r'http\S+\s', ' ', txt)
    txt = re.sub(r'RT|cc', ' ', txt)
    txt = re.sub(r'#\S+\s', ' ', txt)
    txt = re.sub(r'@\S+', ' ', txt)
    txt = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt

def pdf_to_text(file):
    try:
        return extract_text(file)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# ----------------------------
# 3️⃣ Feature Extraction Functions
# ----------------------------
def extract_email(text):
    match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
    return match.group() if match else "Not found"

def extract_contact_number(text):
    match = re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text)
    return match.group() if match else "Not found"

def extract_skills(text):
    skill_list = ['Python', 'Machine Learning', 'Deep Learning', 'SQL', 'Pandas', 'NumPy',
                  'Scikit-learn', 'TensorFlow', 'Tableau', 'Power BI', 'Excel', 'Java', 'C++',
                  'Flask', 'Django', 'REST API', 'Docker', 'Kubernetes']
    skills_found = [skill for skill in skill_list if re.search(r'\b{}\b'.format(skill), text, re.IGNORECASE)]
    return skills_found if skills_found else ["Not found"]

def extract_education(text):
    edu_keywords = ['B.Tech', 'Bachelor', 'M.Tech', 'Master', 'PhD', 'MSc', 'BSc']
    lines = text.split('\n')
    education = [line.strip() for line in lines if any(k.lower() in line.lower() for k in edu_keywords)]
    return education if education else ["Not found"]

def extract_name(text):
    match = re.search(r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)", text)
    return match.group() if match else "Not found"

# ----------------------------
# 4️⃣ Prediction Functions
# ----------------------------
def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    X = tfidf_vectorizer_categorization.transform([resume_text])
    return rf_classifier_categorization.predict(X)[0]

def recommend_job(resume_text):
    resume_text = cleanResume(resume_text)
    X = tfidf_vectorizer_job_recommendation.transform([resume_text])
    return rf_classifier_job_recommendation.predict(X)[0]

# ----------------------------
# 5️⃣ Streamlit UI Styling
# ----------------------------
st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="centered")

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f0f2f6, #d9e4f5);
        color: #333333;
    }
    .title {
        color: #1f4e79;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    .card {
        background-color: #ffffffcc;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .card h4 {
        color: #1f4e79;
        margin-bottom: 10px;
    }
    .card p {
        font-size: 1rem;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📄 AI Resume Screening & Job Recommendation</div>', unsafe_allow_html=True)
st.write("Upload a resume (PDF or TXT) and get predicted category, recommended job, and extracted details.")

# ----------------------------
# 6️⃣ File Upload & Results
# ----------------------------
uploaded_file = st.file_uploader("Choose a resume", type=['pdf', 'txt'])

if uploaded_file is not None:
    # Extract text
    if uploaded_file.name.endswith('.pdf'):
        text = pdf_to_text(uploaded_file)
    else:
        text = uploaded_file.read().decode('utf-8')

    if text.strip() == "":
        st.warning("⚠️ Could not extract text from this file. Try a different PDF.")
    else:
        # Predictions
        category = predict_category(text)
        job = recommend_job(text)
        name = extract_name(text)
        email = extract_email(text)
        phone = extract_contact_number(text)
        skills = extract_skills(text)
        education = extract_education(text)

        # Display results
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"<h4>✅ Predicted Category:</h4><p>{category}</p>", unsafe_allow_html=True)
        st.markdown(f"<h4>💼 Recommended Job:</h4><p>{job}</p>", unsafe_allow_html=True)
        st.markdown(f"<h4>👤 Name:</h4><p>{name}</p>", unsafe_allow_html=True)
        st.markdown(f"<h4>📞 Phone:</h4><p>{phone}</p>", unsafe_allow_html=True)
        st.markdown(f"<h4>📧 Email:</h4><p>{email}</p>", unsafe_allow_html=True)
        st.markdown(f"<h4>🛠 Skills:</h4><p>{', '.join(skills)}</p>", unsafe_allow_html=True)
        st.markdown(f"<h4>🎓 Education:</h4><p>{', '.join(education)}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


