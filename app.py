import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sonar Rock vs Mine Detector",
    page_icon="🌊",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_FILE = "sonar_xgboost_model.pkl"
SCALER_FILE = "sonar_scaler.pkl"

if not os.path.exists(MODEL_FILE):
    st.error(
        "❌ Model file not found!\n\n"
        "Make sure 'sonar_xgboost_model.pkl' is in the same "
        "folder as app.py."
    )
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Load scaler if it exists
scaler = None

if os.path.exists(SCALER_FILE):
    try:
        scaler = joblib.load(SCALER_FILE)
    except Exception:
        scaler = None


# ============================================================
# HEADER
# ============================================================

st.title("🌊 AI-Based Sonar Rock & Mine Detection")

st.subheader("Machine Learning Powered Underwater Object Classification")

st.markdown(
    """
    This system uses a trained **XGBoost Machine Learning model**
    to classify underwater sonar signals as either:

    🪨 **Rock**  
    💣 **Mine**

    The model analyzes **60 numerical sonar frequency features**
    obtained from the reflected sonar signal.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📘 About the Project")

st.sidebar.markdown(
    """
    ### Objective

    Automatically classify sonar signals into
    **Rock** or **Mine**.

    ### Dataset

    - 60 sonar frequency features
    - Binary classification
    - Classes: Rock and Mine

    ### Model

    **XGBoost Classifier**

    ### Evaluation

    The trained model achieved approximately
    **86% test accuracy**.

    ### Explainability

    **SHAP** was used to understand feature importance.
    """
)

st.sidebar.divider()

st.sidebar.info(
    "Enter sonar measurements and click "
    "'Predict' to classify the signal."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📡 Sonar Signal Input")

st.write(
    "Enter the 60 sonar frequency measurements below."
)

input_values = []

# Create 3 columns
columns = st.columns(3)

for i in range(60):

    with columns[i % 3]:

        value = st.number_input(
            f"Frequency Feature {i + 1}",
            value=0.0,
            step=0.001,
            format="%.3f"
        )

        input_values.append(value)


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔍 Predict Rock / Mine",
    use_container_width=True
):

    # Convert input into NumPy array
    input_data = np.array(input_values).reshape(1, -1)

    # Create DataFrame
    input_df = pd.DataFrame(
        input_data,
        columns=[
            f"Frequency Feature {i + 1}"
            for i in range(60)
        ]
    )

    # Apply scaler only if one was saved
    if scaler is not None:
        input_for_prediction = scaler.transform(input_df)
    else:
        input_for_prediction = input_df

    # Make prediction
    prediction = model.predict(input_for_prediction)[0]

    # Get probabilities
    probabilities = model.predict_proba(
        input_for_prediction
    )[0]

    # ========================================================
    # HANDLE MODEL CLASS LABELS
    # ========================================================

    classes = list(model.classes_)

    # Find probability corresponding to prediction
    prediction_index = classes.index(prediction)

    confidence = probabilities[prediction_index] * 100

    # Convert prediction to readable label
    prediction_string = str(prediction).lower()

    if prediction_string in ["1", "m", "mine"]:
        result = "MINE"
        icon = "💣"

    elif prediction_string in ["0", "r", "rock"]:
        result = "ROCK"
        icon = "🪨"

    else:
        # Fallback if model uses different labels
        result = str(prediction).upper()
        icon = "🎯"


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.header("🎯 Prediction Result")

    if result == "MINE":

        st.error(
            f"💣 MINE DETECTED"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.warning(
            "The sonar signal has been classified "
            "as a possible mine."
        )

    elif result == "ROCK":

        st.success(
            f"🪨 ROCK DETECTED"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.info(
            "The sonar signal has been classified "
            "as a rock."
        )

    else:

        st.info(
            f"{icon} Prediction: {result}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )


    # ========================================================
    # PROBABILITY DISPLAY
    # ========================================================

    st.subheader("📊 Prediction Probabilities")

    # Find Rock and Mine probabilities safely
    rock_probability = 0
    mine_probability = 0

    for i, cls in enumerate(classes):

        cls_string = str(cls).lower()

        if cls_string in ["0", "r", "rock"]:
            rock_probability = probabilities[i] * 100

        elif cls_string in ["1", "m", "mine"]:
            mine_probability = probabilities[i] * 100


    probability_df = pd.DataFrame(
        {
            "Class": ["Rock", "Mine"],
            "Probability (%)": [
                rock_probability,
                mine_probability
            ]
        }
    )

    st.bar_chart(
        probability_df.set_index("Class")
    )

    # Display percentages
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "🪨 Rock Probability",
            f"{rock_probability:.2f}%"
        )

    with col2:
        st.metric(
            "💣 Mine Probability",
            f"{mine_probability:.2f}%"
        )


# ============================================================
# CSV UPLOAD
# ============================================================

st.divider()

st.header("📂 Batch Prediction Using CSV")

st.write(
    """
    You can also upload a CSV file containing sonar measurements.
    The first **60 columns** will be treated as sonar features.
    """
)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        uploaded_data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data")

        st.dataframe(
            uploaded_data,
            use_container_width=True
        )

        # Check number of columns
        if uploaded_data.shape[1] < 60:

            st.error(
                "❌ The CSV must contain at least "
                "60 feature columns."
            )

        else:

            # Take first 60 columns
            uploaded_features = uploaded_data.iloc[:, :60]

            # Apply scaler if available
            if scaler is not None:

                uploaded_features = scaler.transform(
                    uploaded_features
                )

            # Predictions
            uploaded_predictions = model.predict(
                uploaded_features
            )

            uploaded_probabilities = model.predict_proba(
                uploaded_features
            )

            st.subheader("🎯 Batch Predictions")

            results = []

            classes = list(model.classes_)

            for i, prediction in enumerate(
                uploaded_predictions
            ):

                prediction_index = classes.index(
                    prediction
                )

                confidence = (
                    uploaded_probabilities[i][prediction_index]
                    * 100
                )

                prediction_string = str(
                    prediction
                ).lower()

                if prediction_string in [
                    "1", "m", "mine"
                ]:

                    label = "💣 MINE"

                elif prediction_string in [
                    "0", "r", "rock"
                ]:

                    label = "🪨 ROCK"

                else:

                    label = str(
                        prediction
                    ).upper()

                results.append(
                    {
                        "Sample": i + 1,
                        "Prediction": label,
                        "Confidence (%)": round(
                            confidence, 2
                        )
                    }
                )

            results_df = pd.DataFrame(results)

            st.dataframe(
                results_df,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"❌ Error processing CSV: {e}"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header("🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Input Features",
        "60"
    )

with col2:

    st.metric(
        "Machine Learning Model",
        "XGBoost"
    )

with col3:

    st.metric(
        "Problem Type",
        "Binary Classification"
    )


# ============================================================
# HOW THE SYSTEM WORKS
# ============================================================

st.subheader("⚙️ How the System Works")

st.markdown(
    """
    **1️⃣ Sonar Signal Input**

    The system receives 60 numerical measurements
    representing the characteristics of a sonar signal.

    **2️⃣ Preprocessing**

    The input is processed using the same preprocessing
    applied during model development.

    **3️⃣ XGBoost Classification**

    The trained XGBoost classifier analyzes the
    sonar pattern.

    **4️⃣ Prediction**

    The system determines whether the signal represents:

    - 🪨 Rock
    - 💣 Mine

    **5️⃣ Confidence Score**

    The model also provides the probability of each class.
    """
)


# ============================================================
# PROJECT PIPELINE
# ============================================================

st.subheader("🔄 Project Pipeline")

st.code(
    """
Sonar Signal
     ↓
60 Frequency Features
     ↓
Data Preprocessing
     ↓
Machine Learning Models
     ↓
XGBoost Classifier
     ↓
Prediction
     ↓
Rock / Mine
     ↓
Confidence Score
     ↓
Streamlit Application
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Based Sonar Rock & Mine Detection | "
    "XGBoost + SHAP + Streamlit"
)