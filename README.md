# Student Performance Prediction System

Production-grade machine learning system for predicting student exam scores using ensemble gradient boosting methods.

**Live Application:** [student-performance-prediction](https://student-performance-predictiongit.streamlit.app/)

---

## Problem Statement

Predict student exam performance (0-100 scale) based on demographic, behavioral, and academic factors. Built as an end-to-end ML pipeline following participation in Kaggle Playground Series S6E1.

**Objective:** Develop a production-ready prediction system with interpretable features and sub-9.0 RMSE performance.

---

## Technical Approach

### Model Architecture

Implemented a weighted ensemble of three gradient boosting algorithms:

- **XGBoost** (Extreme Gradient Boosting) - Level-wise tree growth
- **LightGBM** (Light Gradient Boosting Machine) - Histogram-based, leaf-wise growth
- **CatBoost** (Categorical Boosting) - Ordered boosting for categorical features

**Ensemble Strategy:** Weighted averaging with optimized coefficients based on validation performance.

### Feature Engineering

Derived 7 engineered features from 11 base inputs:

| Feature | Definition | Rationale |
|---------|------------|-----------|
| `study_efficiency` | study_hours × class_attendance | Captures combined effect of time investment and consistency |
| `sleep_study_balance` | sleep_hours / (study_hours + ε) | Models rest-to-study ratio |
| `effort_score` | 2×study_hours + 0.5×attendance + 3×sleep_quality | Weighted composite of key factors |
| `difficulty_penalty` | exam_difficulty / (study_hours + 1) | Accounts for preparedness vs. challenge |
| `learning_support` | internet_access + facility_rating | Infrastructure availability index |
| `cognitive_load` | study_hours × exam_difficulty | Workload intensity metric |
| `recovery_score` | sleep_hours × sleep_quality | Sleep effectiveness measure |

**Total Feature Space:** 18 features (11 original + 7 engineered)

### Data Preprocessing

- **Categorical Encoding:** One-hot encoding for nominal variables (gender, course, study_method)
- **Ordinal Encoding:** Integer mapping for ordinal features (sleep_quality, facility_rating, exam_difficulty)
- **Missing Value Handling:** Zero imputation for derived features (division by epsilon for numerical stability)
- **Scaling:** Not required (tree-based ensemble methods)

### Training & Validation

- **Dataset:** 630,000 training samples from Kaggle Playground Series S6E1
- **Validation Strategy:** 5-fold cross-validation with stratified splits
- **Evaluation Metric:** Root Mean Squared Error (RMSE)
- **Final Performance:** RMSE = 8.71 on holdout test set

### Hyperparameter Optimization

Key parameters tuned via grid search and cross-validation:

- Learning rate: 0.01-0.1
- Max depth: 5-9
- Number of estimators: 500-2000
- Subsample ratio: 0.7-0.9

---

## Project Structure

```
student-performance-predictor/
│
├── notebooks/
│   └── xgb-lgb-catb-ensemble-model.ipynb    # Model development & training
├── app.py              # Main Streamlit application
├── model.pkl           # Trained ensemble model (user-provided)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusions
└── README.md                   # This file
```

---

## Technology Stack

**Core ML Libraries:**
- scikit-learn 1.3.0 (preprocessing, validation)
- xgboost 2.0.3
- lightgbm 4.1.0
- catboost 1.2.2

**Application Framework:**
- Streamlit 1.28.0 (web interface)
- Plotly 5.17.0 (interactive visualizations)

**Development:**
- pandas 2.0.3, numpy 1.24.3
- Python 3.9+

---

## How to Run Locally

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

```bash
# Clone repository
git https://github.com/jamiya-begam-k-17/student-performance-prediction.git
cd student-performance-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Launch Streamlit app
streamlit run streamlit_app/app.py

# Access at http://localhost:8501
```

**Note:** The trained model (`models/ensemble_model.pkl`) is automatically loaded on application startup. No manual model upload is required in the main deployment.

A separate `model_playground` branch includes an experimental model upload feature for testing alternative models. This functionality is intentionally excluded from the main deployment to maintain production stability.

**Branch:** `model_playground`

**Feature:** Manual `.pkl` file upload via Streamlit UI

---

## Deployment

### Production Deployment

The application is deployed on **Streamlit Cloud** with automatic model loading.

**Access:** [https://student-performance-predictiongit.streamlit.app/](https://student-performance-predictiongit.streamlit.app/)

---

## Model Performance

| Metric | Value |
|--------|-------|
| RMSE (Test) | 8.71 |
| RMSE (CV Mean) | 8.73 ± 0.12 |
| Training Samples | 630,000 |
| Test Samples | 270,000 |
| Feature Count | 18 |

### Baseline Comparison

| Model | RMSE |
|-------|------|
| Linear Regression | 12.45 |
| Single XGBoost | 9.02 |
| Single LightGBM | 8.94 |
| Single CatBoost | 9.08 |
| **Weighted Ensemble** | **8.71** |

---

## Key Features

**Input Variables:**
- Demographics: age, gender
- Academic: course, study_method, internet_access
- Behavioral: study_hours, class_attendance, sleep_hours
- Quality: sleep_quality, facility_rating, exam_difficulty

**Output:**
- Predicted exam score (0-100)
- Performance category (Excellent, Good, Needs Improvement, etc.)
- Percentile ranking
- Personalized improvement recommendations

**User Interface:**
- Real-time prediction with interactive form
- Visual gauge chart for score display
- Feature importance analysis
- Improvement projections based on behavioral changes

---

## Engineering Decisions

### Why Ensemble Over Single Model?

**Diversity:** Each algorithm has different strengths—XGBoost handles structured data well, LightGBM offers speed, CatBoost excels with categorical features.

**Variance Reduction:** Weighted averaging reduces prediction variance without significant bias increase.

**Production Stability:** Ensemble approach provides more robust predictions across diverse input distributions.

### Why Tree-Based Methods?

- Naturally handle non-linear relationships
- No feature scaling required
- Built-in feature importance
- Robust to outliers
- Efficient with categorical variables

### Why Streamlit for Deployment?

- Rapid prototyping to production
- Native Python integration
- Built-in state management
- Free cloud hosting
- Automatic HTTPS and deployment

---

## Future Enhancements

**Model Improvements:**
- Bayesian hyperparameter optimization
- Neural network ensemble component
- SHAP values for prediction explainability

**System Enhancements:**
- REST API with FastAPI
- Batch prediction endpoint
- Model versioning and A/B testing
- PostgreSQL integration for prediction logging

**Monitoring:**
- Prediction distribution tracking
- Data drift detection
- Model performance degradation alerts

---

## Limitations

- Model trained on synthetic Kaggle dataset—real-world validation required
- Point predictions only (no confidence intervals)
- Limited to tabular input (no text or time-series features)
- Single-point predictions (no longitudinal tracking)

---

## References

**Data Source:**
Kaggle Playground Series S6E1
[https://www.kaggle.com/competitions/playground-series-s6e1](https://www.kaggle.com/competitions/playground-series-s6e1)

**Documentation:**
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## License

MIT License

## Contact

For technical inquiries or collaboration:

**Developer:** Jamiya Begam K
**Email:** jamiyabegamk@gmail.com

**LinkedIn:** [jamii17](https://www.linkedin.com/in/jamii17/)

---

**Last Updated:** Feb 2025
**Version:** 1.0.0
