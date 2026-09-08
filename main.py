import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="JobVision",
    page_icon="💼"
)

# Load Dataset
df = pd.read_csv("recruitment_data.csv")

with st.sidebar:
    selected = option_menu(
        menu_title="Job Vision",
        options=["Home", "Dataset", "Prediction", "About"],
        icons=["house-fill", "table", "graph-up-arrow", "info-circle-fill"], default_index=0,
        menu_icon="briefcase-fill"
    )

# ----------------Home--------------------
if selected == "Home":
    st.title("JobVision: Intelligent Talent Screening System")
    st.write("""
    This AI system predicts whether a candidate should be hired based on
    their profile.

    - Age

    - Education

    - Experience

    - Interview Score

    - Skill Score

    - Personality Score

    - Recruitment Strategy

    """)
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Candidates", len(df))
    col2.metric("Selected", df["HiringDecision"].sum())
    col3.metric("Rejected", len(df) - df["HiringDecision"].sum())

# ----------------------DataSet---------------------------
elif selected == "Dataset":
    st.title("Dataset Overview")
    st.dataframe(df)

    st.subheader("Dataset Shape")
    st.write("Rows :", df.shape[0])
    st.write("Columns :", df.shape[1])

    st.subheader("Stastical Summary:")
    st.dataframe(df.describe())

# ----------------------Prediction------------------------
elif selected == "Prediction":
    st.title("Hiring Decision Prediction")

    # Feature and Target
    X = df[[
        "Age",
        "Gender",
        "EducationLevel",
        "ExperienceYears",
        "PreviousCompanies",
        "DistanceFromCompany",
        "InterviewScore",
        "SkillScore",
        "PersonalityScore",
        "RecruitmentStrategy"
    ]]

    y = df["HiringDecision"]

    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model call
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"Accuracy : {accuracy:.2%}")

    # Custom Prediction
    st.subheader("Required Details For Hiring:")
    age = st.number_input("Age", 18, 50, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    if gender == "Male":
        gender = 1
    else:
        gender = 0

    education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
    if education == "High School":
        education = 1
    elif education == "Bachelor":
        education = 2
    elif education == "Master":
        education = 3
    else:
        education = 4

    experience = st.number_input("Experience Years", 0, 30, 2)
    previous_companies = st.number_input("Previous Companies", 0, 20, 1)
    distance = st.number_input("Distance from Company", 0.0, 100.0, 10.0)
    interview_score = st.slider("Interview Score", 0, 100, 70)
    skill_score = st.slider("Skill Score", 0, 100, 75)
    personality_score = st.slider("Personality Score", 0, 100, 75)
    strategy = st.selectbox("Recruitment Strategy", [1, 2, 3])
    if st.button("Predict Hiring"):
        data = [[
            age,
            gender,
            education,
            experience,
            previous_companies,
            distance,
            interview_score,
            skill_score,
            personality_score,
            strategy
        ]]
        prediction = model.predict(data)
        if prediction[0] == 1:
            st.success("Candidate is Recommended for Hiring")
        else:
            st.error("Candidate is not Recommended for Hiring")

# ------------------About-----------------------
elif selected == "About":
    st.title("About Project")
    st.subheader("Project Description")
    st.write("""
    JobVision is an AI-ML based recruitment prediction system that analyzes candidate profiles, skills, experience, and qualifications
    to identify suitable applicants for specific job roles. The platform uses intelligent prediction models to
    support faster and more accurate hiring decisions.
    """)
    st.subheader("Machine Learning Algorithm : Random Forest Classifier")
    st.write("""
    The Random Forest Classifier is a supervised machine learning algorithm that operates as an ensemble of decision trees. 
    It builds multiple decision trees during training, each using a random subset of the dataset and features
    (a process known as bootstrap sampling and feature randomness).
    For classification tasks, each tree outputs a class prediction, and the final prediction is determined by majority 
    voting across all trees. This approach reduces overfitting, improves accuracy, and handles both linear and nonlinear
    relationships effectively.
    """)

    st.subheader("Tools and Technologies Uesd:")
    st.write("""
    -Python

    -Pandas

    -Streamlit

    -Scikit-Learn

    -Matplotlib

    -Streamlit Option Menu
    """)

    st.subheader("Developed by")
    st.write("Jaival Bhavsar")
    st.write("Enrollment No. : 246170316007")
    st.write("College Name : Government Polytechnic, Ahmedabad")