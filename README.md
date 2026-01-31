# Student Performance Prediction System

Built an end-to-end machine learning pipeline using XGBoost, LightGBM, and CatBoost with weighted ensembling to predict student performance score.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Input Features](#input-features)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## Overview

This application predicts student exam performance based on various demographic, academic, and behavioral factors. It uses an ensemble approach combining three powerful gradient boosting algorithms to achieve high prediction accuracy (RMSE: 8.72 on test set).

**Key Highlights:**
- Interactive web interface with professional UI/UX
- Real-time score predictions with personalized recommendations
- Visual performance gauge and improvement projections
- Feature engineering with 18 total features (11 base + 7 engineered)
- Trained on 630,000+ student records

## Features

### Core Functionality
- **Ensemble ML Model**: Combines XGBoost, LightGBM, and CatBoost for robust predictions
- **Interactive Web Interface**: Clean, professional Streamlit-based UI
- **Real-time Predictions**: Instant score predictions with visual feedback
- **Personalized Recommendations**: Actionable insights based on student profile
- **Performance Visualization**: Gauge charts and score breakdowns
- **Improvement Projections**: Shows potential score improvements

### Technical Features
- Custom CSS styling for professional appearance
- Session state management for smooth user experience
- Input validation and error handling
- Feature engineering pipeline
- One-hot and ordinal encoding
- Responsive design with multi-column layouts

## Project Structure

```
student-performance-predictor/
│
└── notebooks
|    └── xgb-lgb-catb-ensemble-model.ipynb      # Jupyter notebook
├── app.py              # Main Streamlit application
├── model.pkl           # Trained ensemble model (user-provided)
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation

```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/student-performance-predictor.git
   cd student-performance-predictor
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **Start the Streamlit server**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - The app will automatically open in your default browser

### Making Predictions

1. **Upload Model**
   - Upload the `model.pkl` file through the sidebar interface
   - You can use the pkl file in the repository or you can train your self a pkl file using the notebook I attached in notebooks with any other techniques or models.

2. **Enter Student Information**
   - Fill in all required fields in the form:
     - Demographics (age, gender, internet access)
     - Academic info (course, study method)
     - Study habits (study hours, attendance, sleep hours)
     - Quality indicators (sleep quality, facilities, exam difficulty)

3. **Get Prediction**
   - Click "Predict Score" button
   - View results including:
     - Predicted score with performance gauge
     - Score breakdown and category
     - Personalized recommendations
     - Improvement projections

## Model Details

### Ensemble Architecture
The prediction system uses a weighted ensemble of three gradient boosting algorithms:

- **XGBoost**: Level-wise tree growth, excellent for structured data
- **LightGBM**: Leaf-wise growth, fast training and inference
- **CatBoost**: Ordered boosting, handles categorical features well

### Performance Metrics
- **RMSE**: 8.72 (on test set)
- **Training Data**: 630,000 samples
- **Validation**: 5-fold cross-validation
- **Source**: Kaggle Playground Series S6E1

### Feature Engineering
The model uses 18 features total:

**11 Base Features:**
- age, gender, course, study_method
- internet_access, study_hours, class_attendance
- sleep_hours, sleep_quality, facility_rating, exam_difficulty

**7 Engineered Features:**
1. `study_efficiency` = study_hours × class_attendance
2. `sleep_study_balance` = sleep_hours / (study_hours + 0.1)
3. `effort_score` = (study_hours × 2) + (attendance × 0.5) + (sleep_quality × 3)
4. `difficulty_penalty` = exam_difficulty / (study_hours + 1)
5. `learning_support` = internet_access + facility_rating
6. `cognitive_load` = study_hours × exam_difficulty
7. `recovery_score` = sleep_hours × sleep_quality


## Screenshots

### Main Interface
![alt text](assets\image.png)

### Prediction Results
![alt text](assets\image-1.png)

### Improvement Projections
![alt text](assets\image-2.png)

## Customization

### Styling
Modify the `load_css()` function in `app.py` to customize:
- Color schemes
- Font styles
- Layout spacing
- Component styling

### Model
Replace `model.pkl` with your own trained model. Ensure it:
- Accepts the same 18 features
- Returns predictions in range [0, 100]
- Is pickled using the same scikit-learn/XGBoost/LightGBM/CatBoost versions

### Recommendations
Edit the `generate_recommendations()` function to customize the advice logic.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: Kaggle Playground Series S6E1 [https://www.kaggle.com/competitions/playground-series-s6e1/data]
- Libraries: Streamlit, Scikit-learn, XGBoost, LightGBM, CatBoost, Plotly
- Community: Thanks to all contributors and users

## Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ using Streamlit and Ensemble ML**
