# CardioSense — Heart Disease Prediction System

A full-stack medical AI web application for heart disease risk prediction using four machine learning models.

---

## Project Structure

```
heart_app/
│
├── app.py                  ← Flask backend (ML models + API)
├── requirements.txt        ← Python dependencies
│
├── templates/
│   └── index.html          ← Main HTML (single-page app)
│
└── static/
    ├── css/
    │   └── style.css       ← Clean white medical UI styles
    └── js/
        └── app.js          ← Frontend logic & API calls
```

---

## Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
python app.py
```

### 3. Open in Browser
```
http://localhost:5050
```

---

## Features

| Feature | Description |
|---|---|
| **Predict** | Enter patient data, get probability scores from all 4 models |
| **Dashboard** | Model accuracy, AUC-ROC, F1-score comparison table |
| **Analytics** | Feature importance, ROC curves, confusion matrices, probability distributions |
| **Responsive** | Works on desktop and mobile |

---

## ML Models

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 81.95% | 0.9017 |
| **Random Forest** | **89.76%** | **0.9443** |
| SVM | 82.93% | 0.9108 |
| KNN | 77.56% | 0.8279 |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main web UI |
| `/api/stats` | GET | Dataset stats & model metrics |
| `/api/predict` | POST | Predict from patient JSON |
| `/api/chart/roc` | GET | ROC curve chart (base64 PNG) |
| `/api/chart/accuracy_bar` | GET | Accuracy/AUC bar chart |
| `/api/chart/confusion` | GET | Confusion matrices |
| `/api/chart/feature_importance` | GET | RF feature importance |
| `/api/chart/prob_dist` | GET | Probability distributions |

---

## Dataset

- **Source**: Kaggle Heart Disease Dataset (Cleveland UCI)
- **Records**: 1,025 patients
- **Features**: 13 clinical parameters
- **Target**: Binary (Heart Disease: 1 / No Disease: 0)
- **Split**: 80% train / 20% test (Stratified)
- **Validation**: 5-Fold Stratified Cross-Validation
