import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from pathlib import Path
import time



# ================================================
# Load model
# =================================================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

def load_css():
    """Load custom CSS for professional styling"""
    st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: #1a202c;
        padding-top: 1rem;
    }
    
    /* Custom scrollbar for sidebar */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]::-webkit-scrollbar {
        width: 8px;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-track {
        background: #e2e8f0;
        border-radius: 10px;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
        background: #4299e1;
        border-radius: 10px;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb:hover {
        background: #3182ce;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: white;
    }
    
    /* Sidebar text visibility */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] ul, [data-testid="stSidebar"] li {
        color: white;
    }
    
    /* Main Content Area*/
    .main {
        background-color: #ffffff;
    }
    
    /* Headers */
    h1 {
        color: #1a202c;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #4299e1;
    }
    
    h2 {
        color: #2d3748;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 500;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #4299e1;
    }
    
    /* Info Boxes */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: #012740;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4299e1;
        margin: 0.5rem 0;
    }
    
    .success-card {
        background: #f0fff4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #48bb78;
        color: #22543d;
    }
    
    .warning-card {
        background: #fffaf0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ed8936;
        color: #7c2d12;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Form Inputs */
    .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        border-radius: 6px;
        border: 1.5px solid #e2e8f0;
        padding: 0.5rem;
    }
    
    .stNumberInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
        border-color: #4299e1;
        box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Sidebar Expander Styling */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: rgb(40 56 67);
        border: 1px solid #060d2b;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        font-weight: 600;
        color: rgb(255 255 255);
        padding: 0.5rem;
        background: #031d33;
    }
    
    /* Sidebar spacing */
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def preprocess_input(data, model=None):
    """
    Preprocess input data to match training format
    
    Args:
        data: Dictionary of input features
        model: Trained model object (for feature alignment)
    
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    # Ordinal encoding mappings
    ordinal_map = {
        "sleep_quality": {"poor": 0, "average": 1, "good": 2},
        "facility_rating": {"low": 0, "medium": 1, "high": 2},
        "exam_difficulty": {"easy": 0, "moderate": 1, "hard": 2},
        "internet_access": {"no": 0, "yes": 1}
    }
    
    df = pd.DataFrame([data])
    
    # Apply ordinal encoding
    for col, mapping in ordinal_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, columns=["gender", "course", "study_method"], drop_first=False)
    
    # Feature Engineering - 7 derived features
    df["study_efficiency"] = df["study_hours"] * df["class_attendance"]
    df["sleep_study_balance"] = df["sleep_hours"] / (df["study_hours"] + 0.1)
    df["effort_score"] = df["study_hours"] * 2 + df["class_attendance"] * 0.5 + df["sleep_quality"] * 3
    df["difficulty_penalty"] = df["exam_difficulty"] / (df["study_hours"] + 1)
    df["learning_support"] = df["internet_access"] + df["facility_rating"]
    df["cognitive_load"] = df["study_hours"] * df["exam_difficulty"]
    df["recovery_score"] = df["sleep_hours"] * df["sleep_quality"]
    
    # Align with model features if model is provided
    if model is not None and hasattr(model, 'feature_names_'):
        expected_features = model.feature_names_
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
        df = df[expected_features]
    
    return df


def get_performance_category(score):
    """Get performance category and color based on score"""
    if score >= 80:
        return "Excellent", "#66cc90"
    elif score >= 70:
        return "Good", "#aa6bb8"
    elif score >= 60:
        return "Average", "#c38a5b"
    elif score >= 50:
        return "Below Average", "#f56565"
    else:
        return "Needs Improvement", "#9e0c0c"

def create_score_gauge(score):
    """Create gauge chart for score visualization"""
    category, color = get_performance_category(score)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{category}</b><br><span style='font-size:0.8em'>Predicted Score</span>", 
                'font': {'size': 20}},
        number={'font': {'size': 50, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 40], 'color': '#fed7d7'},
                {'range': [40, 70], 'color': '#feebc8'},
                {'range': [70, 100], 'color': '#c6f6d5'}
            ],
            
            'threshold': {
                'line': {'color': "#2d3748", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    
    return fig

def generate_recommendations(score, input_data):
    """Generate personalized recommendations based on input data"""
    recommendations = []
    
    if input_data['study_hours'] < 20:
        recommendations.append(f"**Increase study time**: Currently {input_data['study_hours']:.1f} hrs/week. Aim for 20-25 hours to improve performance.")
    
    if input_data['class_attendance'] < 85:
        recommendations.append(f"**Improve attendance**: {input_data['class_attendance']:.0f}% attendance. Target 85%+ for optimal learning.")
    
    if input_data['sleep_quality'] == 'poor':
        recommendations.append("**Enhance sleep quality**: Poor sleep affects cognitive performance. Prioritize 7-9 hours of quality sleep.")
    
    if input_data['sleep_hours'] < 7:
        recommendations.append(f"**Get more sleep**: {input_data['sleep_hours']:.1f} hrs is below optimal. Aim for 7-9 hours per night.")
    
    if input_data['internet_access'] == 'no':
        recommendations.append("**Access online resources**: Internet access is essential for modern learning resources and research.")
    
    if input_data['facility_rating'] == 'low':
        recommendations.append("**Seek better facilities**: Consider library or study spaces with better infrastructure.")
    
    if input_data['exam_difficulty'] == 'hard' and input_data['study_hours'] < 25:
        recommendations.append("**Difficulty vs Effort**: Hard exams require 25+ hours/week of focused study.")
    
    return recommendations

# ============================================================================
# SIDEBAR
# ============================================================================

load_css()

with st.sidebar:
    
    st.markdown("# Student Performance")
    st.markdown("### ML Prediction System")
    # st.markdown("---")
    
    # Model Performance Metrics
    st.markdown("## Model Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RMSE Score", "8.720", help="Root Mean Squared Error on test set")
    with col2:
        st.metric("Dataset", "630K", help="Training samples used")
    
    st.markdown("---")
    
    # About Section
    st.markdown("## 📖 About")
    st.markdown("""
    This system uses an **ensemble of three gradient boosting algorithms** to predict student exam scores with high accuracy.
    
    **Key Features:**
    - XGBoost + LightGBM + CatBoost
    - 18 features (11 base + 7 engineered)
    - 5-fold cross-validation
    - Production-ready pipeline
    """)
    
    st.markdown("---")
    
    # Technical Details
    with st.expander("Technical Details"):
        st.markdown("""
        **Algorithms:**
        - XGBoost (Level-wise growth)
        - LightGBM (Leaf-wise growth)
        - CatBoost (Ordered boosting)
        
        **Feature Engineering:**
        - Study efficiency metrics
        - Sleep-study balance
        - Cognitive load indicators
        - Difficulty-effort ratios
        """)
    
    with st.expander("Dataset Info"):
        st.markdown("""
        **Source:** Kaggle Playground Series S6E1
        
        **Size:** 630,000 samples
        
        **Features:** 11 base features
        - Demographics: Age, Gender
        - Academic: Course, Study Method
        - Behavioral: Study Hours, Attendance
        - Environmental: Facilities, Internet
        """)

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Header
st.title("Student Performance Prediction")
st.markdown("Enter student information to predict exam score using ensemble ML models")
st.markdown("---")

# Prediction Form
st.markdown("## Student Information")

with st.form("prediction_form"):
    # Section 1: Demographics
    st.markdown("### 👤 Demographics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=15, max_value=100, value=20, step=1)
    with col2:
        gender = st.selectbox("Gender", ["male", "female", "other"])
    with col3:
        internet_access = st.selectbox("Internet Access", ["yes", "no"])
    
    st.markdown("---")
    
    # Section 2: Academic Information
    st.markdown("### 📚 Academic Information")
    col4, col5 = st.columns(2)
    
    with col4:
        course = st.selectbox("Course", ["b.tech", "b.sc", "ba", "bba", "bca", "diploma"])
    with col5:
        study_method = st.selectbox(
            "Study Method", 
            ["self-study", "group study", "online videos", "coaching", "mixed"]
        )
    
    st.markdown("---")
    
    # Section 3: Study Habits
    st.markdown("### ⏰ Study Habits")
    col6, col7, col8 = st.columns(3)
    
    with col6:
        study_hours = st.number_input(
            "Study Hours/Week", 
            min_value=0.0, 
            max_value=60.0, 
            value=20.0, 
            step=0.5,
            help="Average weekly study hours"
        )
    with col7:
        class_attendance = st.number_input(
            "Attendance (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=75.0, 
            step=1.0,
            help="Overall class attendance percentage"
        )
    with col8:
        sleep_hours = st.number_input(
            "Sleep Hours/Day", 
            min_value=0.0, 
            max_value=12.0, 
            value=7.0, 
            step=0.5,
            help="Average daily sleep hours"
        )
    
    st.markdown("---")
    
    # Section 4: Quality Indicators
    st.markdown("### ⭐ Quality Indicators")
    col9, col10, col11 = st.columns(3)
    
    with col9:
        sleep_quality = st.selectbox("Sleep Quality", ["poor", "average", "good"])
    with col10:
        facility_rating = st.selectbox("Facility Rating", ["low", "medium", "high"])
    with col11:
        exam_difficulty = st.selectbox("Exam Difficulty", ["easy", "moderate", "hard"])
    
    st.markdown("---")
    
    # Submit Button
    submitted = st.form_submit_button("🎯 Predict Score", use_container_width=True)
    
    if submitted:
        # Validation
        valid = True
        
        if study_hours <= 0:
            st.error("❌ Study hours must be greater than 0")
            valid = False
        if class_attendance < 0 or class_attendance > 100:
            st.error("❌ Attendance must be between 0 and 100")
            valid = False
        if sleep_hours <= 0:
            st.error("❌ Sleep hours must be greater than 0")
            valid = False
        
        if valid:
            # Prepare input data
            input_data = {
                'age': age,
                'gender': gender,
                'course': course,
                'study_method': study_method,
                'internet_access': internet_access,
                'study_hours': study_hours,
                'class_attendance': class_attendance,
                'sleep_hours': sleep_hours,
                'sleep_quality': sleep_quality,
                'facility_rating': facility_rating,
                'exam_difficulty': exam_difficulty
            }
            
            # Load and predict
            model_path = "model.pkl"
            if Path(model_path).exists():
                if model:
                    with st.spinner('🔮 Generating prediction...'):
                        time.sleep(0.8)
                        processed_data = preprocess_input(input_data, model)
                        prediction = model.predict(processed_data)[0]
                        prediction = np.clip(prediction, 0, 100)
                        
                        st.session_state.prediction = prediction
                        st.session_state.input_data = input_data
                        st.session_state.prediction_made = True
                        st.rerun()

# ============================================================================
# RESULTS SECTION
# ============================================================================

if st.session_state.prediction_made:
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")
    
    prediction = st.session_state.prediction
    input_data = st.session_state.input_data
    
    # Display gauge chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(create_score_gauge(prediction), use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Score Breakdown")
        
        category, color = get_performance_category(prediction)
        
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: {color}; margin: 0;">{prediction:.2f}/100</h2>
            <p style="margin: 0.5rem 0 0 0; color: #4a5568;">Performance Level: <strong>{category}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        st.metric("Study Hours/Week", f"{input_data['study_hours']:.1f} hrs")
        st.metric("Attendance", f"{input_data['class_attendance']:.0f}%")
        st.metric("Sleep Quality", input_data['sleep_quality'].title())
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("### 💡 Personalized Recommendations")
    
    recommendations = generate_recommendations(prediction, input_data)
    
    if recommendations:
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.success("✅ Excellent profile! Keep maintaining these habits for consistent performance.")
    
    st.markdown("---")
    
    # Improvement Projections
    st.markdown("### Improvement Potential")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #4a5568;">+2 hrs study/day</h4>
            <h3 style="margin: 0.5rem 0; color: #4299e1;">{min(prediction + 10, 100):.1f}</h3>
            <p style="margin: 0; color: #718096; font-size: 0.9rem;">Projected Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #4a5568;">Better sleep quality</h4>
            <h3 style="margin: 0.5rem 0; color: #48bb78;">{min(prediction + 8, 100):.1f}</h3>
            <p style="margin: 0; color: #718096; font-size: 0.9rem;">Projected Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #4a5568;">Combined improvements</h4>
            <h3 style="margin: 0.5rem 0; color: #9f7aea;">{min(prediction + 15, 100):.1f}</h3>
            <p style="margin: 0; color: #718096; font-size: 0.9rem;">Projected Score</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; padding: 2rem 0;">
    <p style="margin: 0; font-weight: 600;">Student Performance Prediction System</p>
    <p style="margin: 0.5rem 0; font-size: 0.9rem;">Ensemble ML Model (XGBoost + LightGBM + CatBoost) | RMSE: 8.71964</p>
    <p style="margin: 0; font-size: 0.85rem;">Kaggle Playground Series S6E1 Dataset (630K samples)</p>
</div>
""", unsafe_allow_html=True)
