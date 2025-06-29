
# 💼 Tech Company Resume Screening Bot (Role-Aware Filtering)

An AI-powered(all-MiniLM-L6-v2 model) resume screening app designed for tech companies to identify top candidates for a given job role. It uses semantic similarity, skill and certification matching, education level ranking, and experience normalization to recommend the best applicants. It focuses on the meaning behind words rather than value charachter matching.

The project leverages several powerful Python libraries to build a robust and interpretable resume screening system. Streamlit powers the interactive web interface, allowing users to upload applicant data and visualize analytics effortlessly. Pandas and NumPy handle data manipulation and numerical operations, including experience normalization and feature engineering. SentenceTransformers (via Hugging Face) provides semantic embedding capabilities using the "all-MiniLM-L6-v2 model", enabling intelligent matching between resumes and job descriptions. Seaborn and Matplotlib are used for generating insightful data visualizations such as skill distributions, salary trends, and role frequencies. Additionally, base64 is employed to enable downloading the top candidates’ results directly from the app as a CSV file. Together, these libraries create a full-stack machine learning solution that is both functional and explainable.

⸻

## 🚀 Features
	•	Semantic Screening using the SentenceTransformers model (all-MiniLM-L6-v2)
	•	Role-aware filtering — job role similarity weighted more than generic text similarity
	•	Skill & Certification Relevance Scoring
	•	Education Scoring based on degree hierarchy
	•	Experience Normalization using log1p scaling
	•	Downloadable Top Candidates List
	•	Insightful Analytics:
	•	Salary expectation distribution
	•	Most popular job roles
	•	Most common degrees
	•	Projects count distribution
	•	Top 10 most frequent skills

⸻

# How it Works

1. Data Upload
	•	Upload a CSV of applicants with the following fields:

Name, Skills, Experience (Years), Education, Certifications, Job Role, Salary Expectation ($), Projects Count


	•	Paste the Job Description in the text box.

⸻

# 🧠 Scoring Algorithm

The resume screening model uses a composite scoring system to rank candidates based on how well they match the provided job description. The score is calculated as a weighted sum of five core components:

Job Role Similarity-39.25%	
Measures how closely the applicant’s job role matches the job description.

Skill Similarity-37%
Semantic similarity between listed skills and job requirements.


Certification Match-12.5%
Relevance of certifications to the job description.

Experience-5.625%	
Years of experience (normalized using log scale).

Education Level-5.625%
Degree converted to a ranked score (e.g., PhD > M.Tech > B.Tech, etc.).

## 🔢 Formula

Final Score = 
    0.3925 * Role Similarity +
    0.37 * Skill Similarity +
    0.125 * Certification Similarity +
    0.05625 * Normalized Experience +
    0.05625 * Normalized Education Score

#### 🎓 Education Ranking

The education score is computed using a predefined ranking:

Degree	Score

PhD=5

M.Tech=4

MBA=3

B.Tech=2

B.Sc=1

Anything else 0

Any other degree or unknown entry is scored as 0.

#### 📏 Experience Normalization

To prevent long experience years from disproportionately skewing results, experience is scaled using:

exp_log = np.log1p(applicant_df["Experience (Years)"])

exp_norm = exp_log / exp_log.max()

This ensures fair scaling across candidates with both low and high experience.

⸻

✨ Why These Weights?

	•	Role and Skills are prioritized as they most directly indicate fit.
 
	•	Certifications support credibility.
 
	•	Experience and Education add context but are considered supporting factors.

## Result Output
	•	Top 10 candidates are displayed with all relevant fields.
	•	Downloadable CSV of top candidates.

### Data Visualization

Several visualizations are generated from applicant data:
	•	Salary Distribution Histogram
	•	Bar Charts for Job Role & Degree Frequencies
	•	Project Count Distribution
	•	Top 10 Skills Frequency

⸻

## 🔧 Installation & Run Instructions
Install the following libraries and dependencies

streamlit

pandas

numpy

matplotlib

seaborn

sentence-transformers



⸻

### ✅ Sample Applicant CSV Format

Name,Skills,Experience (Years),Education,Certifications,Job Role,Salary Expectation ($),Projects Count

John Doe,"Python, Machine Learning, SQL",3,B.Tech,"AWS Certified, DataCamp",Data Scientist,60000,5

Jane Smith,"Java, Spring Boot",2,M.Tech,"Oracle Certified",Software Engineer,50000,3


⸻

### 🔍 Example Use Cases
	•	Startups hiring for tech positions with limited HR bandwidth.
	•	Recruiters filtering applicants based on job-specific relevance.
	•	Hackathons and Internship Portals scoring candidates quickly.

⸻

### ✨ Future Enhancements
	•	Integration with LinkedIn scraping or resume parsing
	•	Experience in relevant domain rather than raw years
	•	Better certification ranking via external source (e.g., Coursera, Google, etc.)
	•	UI Enhancements: Multi-role screening, pagination, model explainability

⸻

License

This project is licensed under the MIT License.

⸻

Author

Krishiv Garg

⸻
