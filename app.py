import streamlit as st
import numpy as np
import joblib


st.set_page_config(
    page_title="🔬 Cell Classifier",
    layout="wide",
    page_icon="🧬"
)

model = joblib.load('svm_cell_classifier.joblib')

st.markdown("""
    <style>
    .main {
        background-color: #fafafa;
    }
    .block-container {
        padding-top: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f0f4f8;
        text-align: center;
        font-size: 20px;
    }
    .benign {
        background-color: #d0f0c0;
        color: #2d572c;
        font-weight: bold;
    }
    .malignant {
        background-color: #ffd1d1;
        color: #8b0000;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Cell Sample Classification App (SVM)")
st.markdown("""
Welcome to the **Cell Classifier App** powered by **Support Vector Machine (SVM)**.  
This tool helps predict whether a **cell sample** is likely to be:
- 🟦 **Benign (Class 2)** or  
- 🟥 **Malignant (Class 4)**  
based on 9 cytological characteristics.  
""")

st.markdown("---")

st.sidebar.title("🧪 Input Features")
st.sidebar.markdown("Adjust the values for the sample below:")

feature_labels = [
    "Clump Thickness",
    "Uniformity of Cell Size",
    "Uniformity of Cell Shape",
    "Marginal Adhesion",
    "Single Epithelial Cell Size",
    "Bare Nuclei",
    "Bland Chromatin",
    "Normal Nucleoli",
    "Mitoses"
]

default_values = [5, 5, 5, 5, 5, 5, 5, 5, 1]

user_input = []
for label, default in zip(feature_labels, default_values):
    value = st.sidebar.slider(label, 1, 10, default)
    user_input.append(value)

input_array = np.array(user_input).reshape(1, -1)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔍 Predict Cell Type"):
        prediction = model.predict(input_array)[0]

        if prediction == 2:
            st.success("🟦 The model predicts this sample is **Benign (Class 2)**.")
            st.markdown('<div class="prediction-box benign">🟦 Benign (Non-cancerous)</div>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.error("🟥 The model predicts this sample is **Malignant (Class 4)**.")
            st.markdown('<div class="prediction-box malignant">🟥 Malignant (Potentially Cancerous)</div>', unsafe_allow_html=True)
            st.warning("⚠️ Please consult a medical expert for further testing.")

with st.expander("📖 What do these features mean?"):
    st.markdown("""
    - **Clump Thickness**: Density of cell clusters.
    - **Uniformity of Cell Size**: Consistency in cell size.
    - **Uniformity of Cell Shape**: Consistency in cell shape.
    - **Marginal Adhesion**: Cell stickiness to each other.
    - **Single Epithelial Cell Size**: Size of individual epithelial cells.
    - **Bare Nuclei**: Cells with visible nucleus only.
    - **Bland Chromatin**: Smoothness of DNA material in the nucleus.
    - **Normal Nucleoli**: Appearance of nucleoli inside nucleus.
    - **Mitoses**: Rate of cell division.

    These features are extracted from FNA (Fine Needle Aspiration) test results.
    """)

st.markdown("---")
st.caption("💻 Developed using Streamlit & scikit-learn | Final Project: Classification with SVM")
