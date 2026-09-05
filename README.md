# AI-Powered Sonar Intelligence for Underwater Rock & Mine Detection

## 🚀 Project Overview

This project uses **Artificial Intelligence and Machine Learning** to classify underwater sonar signals as either **Rock** or **Mine**.

The system analyzes **60 numerical frequency-based sonar features** and uses an **XGBoost classification model** to identify the detected object. It also provides prediction probabilities to indicate the model's confidence.

The trained model is deployed through an interactive **Streamlit web application**.

## 🎯 Problem Statement

Underwater environments can be difficult and potentially dangerous to inspect. When sonar detects an unknown object, determining whether it is a harmless rock or a potentially dangerous mine can be challenging.

Our project provides an AI-powered first-level assessment by automatically analyzing sonar signal patterns and classifying the detected object as **Rock or Mine**.

## 💡 Solution

The system follows this pipeline:

**Sonar Data → 60 Features → XGBoost → Rock / Mine → Confidence Score**

The model learns patterns from labeled sonar observations and uses those patterns to classify new observations.

## 🧠 Machine Learning

The final model used is **XGBoost**, selected because it performs well on structured numerical data and provides strong classification performance.

### Model Performance

* **Accuracy:** ~86%

| Class | Precision | Recall | F1-Score |
| ----- | --------: | -----: | -------: |
| Rock  |      0.85 |   0.85 |     0.85 |
| Mine  |      0.86 |   0.86 |     0.86 |

The model was also evaluated using an **ROC curve and ROC-AUC**.

## 🔍 Explainable AI

We used **SHAP (SHapley Additive exPlanations)** to understand which sonar features have the greatest influence on the model's predictions.

This makes the system more interpretable instead of treating the model as a complete black box.

## 🖥️ Application

The trained XGBoost model is integrated into a **Streamlit application**.

The application provides:

* Rock/Mine prediction
* Prediction probability
* Confidence score
* Interactive sonar feature input

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* XGBoost
* Joblib
* Streamlit
* SHAP
* Matplotlib

## 📁 Project Structure

```text
Sonar-Versus-Mine-Detection/
│
├── app.py
├── sonar_xgboost_model.pkl
├── sonar_scaler.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/imshakhan-13/Sonar-Versus-Mine-Detection.git
cd Sonar-Versus-Mine-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🌊 Future Scope

Future improvements could include:

* Training on larger real-world sonar datasets
* Handling underwater noise and changing sensor conditions
* Real-time sonar signal processing
* Integration with Autonomous Underwater Vehicles (AUVs)
* Improved validation for real-world underwater environments

## ⚠️ Disclaimer

This is a machine-learning prototype developed for research and demonstration purposes. The reported performance is based on the evaluation dataset and should not be interpreted as certified real-world mine detection capability.
