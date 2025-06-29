import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
import base64

st.title("Tech Company Resume Screening Bot (Role-Aware Filtering)")

# Upload training data
@st.cache_resource
def load_training_data():
    return pd.read_csv("Resume_Model/data.csv")

data = load_training_data()


def preprocess_data(df):
    df = df.copy()
    df.fillna("", inplace=True)
    df["Combined"] = (
        df["Skills"] + " " +
        df["Education"] + " " +
        df["Certifications"] + " " +
        df["Job Role"]
    )
    return df

# Education ranking mapping
education_rank = {
    "PhD": 5,
    "M.Tech": 4,
    "MBA": 3,
    "B.Tech": 2,
    "B.Sc": 1
}

def get_education_score(education):
    return education_rank.get(education.strip(), 0)

# Load Hugging Face semantic model
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

semantic_model = load_semantic_model()

# Upload applicant CSV
uploaded_file = st.file_uploader("Upload the applicant CSV file", type=["csv"])
job_description = st.text_area("Paste the job description")

if uploaded_file and job_description:
    st.info("Analyzing uploaded file...")
    applicant_df = pd.read_csv(uploaded_file)
    applicant_df.fillna("", inplace=True)

    # Prepare embeddings
    job_embedding = semantic_model.encode(job_description, convert_to_tensor=True)
    applicant_df["Combined"] = (
        applicant_df["Skills"] + " " +
        applicant_df["Education"] + " " +
        applicant_df["Certifications"] + " " +
        applicant_df["Job Role"]
    )

    cert_embeddings = semantic_model.encode(applicant_df["Certifications"].tolist(), convert_to_tensor=True)
    cert_similarities = util.cos_sim(cert_embeddings, job_embedding).cpu().numpy().flatten()

    # Encode applicant skills
    skill_embeddings = semantic_model.encode(applicant_df["Skills"].tolist(), convert_to_tensor=True)
    skill_similarities = util.cos_sim(skill_embeddings, job_embedding).cpu().numpy().flatten()
    # Job role embedding and similarity
    role_embeddings = semantic_model.encode(applicant_df["Job Role"].tolist(), convert_to_tensor=True)
    role_similarities = util.cos_sim(role_embeddings, job_embedding).cpu().numpy().flatten()

    # Normalize experience and map education
    exp_log = np.log1p(applicant_df["Experience (Years)"])  # log1p(x) = log(x+1)
    exp_norm = exp_log / exp_log.max()
    applicant_df["Education Score"] = applicant_df["Education"].apply(get_education_score)
    edu_norm = applicant_df["Education Score"] / max(education_rank.values())

    # Final fair score with emphasis on job role
    applicant_df["Score"] = (
    0.35 * role_similarities +
    0.30 * skill_similarities +
    0.15 * cert_similarities +
    0.10 * exp_norm +
    0.10 * edu_norm 
    )

    top_candidates = applicant_df.sort_values("Score", ascending=False).head(10)

    result_df = top_candidates[[
        "Name", "Job Role", "Skills", "Experience (Years)", "Education",
        "Certifications", "Salary Expectation ($)"
    ]]

    st.success("Top 10 Candidates Identified")
    st.dataframe(result_df)

    # Download link
    csv = result_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="top_candidates.csv">Download Top Candidates CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Analysis Section
    st.subheader("📊 Data Analysis")

    # Salary Distribution
    fig1, ax1 = plt.subplots()
    sns.histplot(applicant_df["Salary Expectation ($)"], bins=10, kde=False, ax=ax1)
    ax1.set_title("Salary Expectation Distribution")
    ax1.set_xlabel("Salary ($)")
    ax1.set_ylabel("Number of Candidates")
    st.pyplot(fig1)

    # Most Popular Job Roles
    fig2, ax2 = plt.subplots()
    role_counts = applicant_df["Job Role"].value_counts().head(10)
    sns.barplot(x=role_counts.values, y=role_counts.index, ax=ax2)
    ax2.set_title("Most Popular Job Roles")
    ax2.set_xlabel("Number of Candidates")
    st.pyplot(fig2)

    # Most Common Degrees
    fig3, ax3 = plt.subplots()
    edu_counts = applicant_df["Education"].value_counts().head(10)
    sns.barplot(x=edu_counts.values, y=edu_counts.index, ax=ax3)
    ax3.set_title("Most Common Degrees")
    ax3.set_xlabel("Number of Candidates")
    st.pyplot(fig3)


    # Project Count Distribution
    st.subheader("Project Count Distribution")
    fig5, ax5 = plt.subplots()
    sns.histplot(applicant_df["Projects Count"], bins=10, kde=False, ax=ax5)
    ax5.set_title("Number of Projects Completed")
    ax5.set_xlabel("Projects Count")
    ax5.set_ylabel("Number of Candidates")
    st.pyplot(fig5)

    # Top Skills
    st.subheader("Top Skills Frequency")
    all_skills = applicant_df["Skills"].str.split(",").explode().str.strip().value_counts().head(10)
    fig6, ax6 = plt.subplots()
    sns.barplot(x=all_skills.values, y=all_skills.index, ax=ax6)
    ax6.set_title("Top 10 Most Frequent Skills")
    ax6.set_xlabel("Frequency")
    st.pyplot(fig6)


