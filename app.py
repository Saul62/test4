import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title="Severe pneumonia in children", layout="wide")

# Load model
@st.cache_resource
def load_model():
    model = load('model.pkl')  # 使用相对路径
    return model

model = load_model()

# Create page title
st.title("Severe pneumonia in children")
st.write("Please input patient's clinical indicators:")

# Create input fields
col1, col2, col3 = st.columns(3)

with col1:
    cl = st.number_input('Chloride (Cl) (mmol/L)', value=100.0, format="%.1f")
    glu = st.number_input('Glucose (GLU) (mmol/L)', value=5.5, format="%.1f")
    dbil = st.number_input('Direct Bilirubin (DBil) (μmol/L)', value=10.0, format="%.1f")
    ldh = st.number_input('Lactate Dehydrogenase (LDH) (U/L)', value=200.0, format="%.1f")

with col2:
    bun_scr = st.number_input('BUN/SCr Ratio', value=20.0, format="%.1f")
    che = st.number_input('Cholinesterase (CHE) (U/L)', value=8000.0, format="%.1f")
    ibil = st.number_input('Indirect Bilirubin (IBil) (μmol/L)', value=8.0, format="%.1f")
    ua = st.number_input('Uric Acid (UA) (μmol/L)', value=300.0, format="%.1f")

with col3:
    pdw = st.number_input('Platelet Distribution Width (PDW) (%)', value=12.0, format="%.1f")
    ggt = st.number_input('Gamma-Glutamyl Transferase (GGT) (U/L)', value=30.0, format="%.1f")
    ly_pct = st.number_input('Lymphocyte Percentage (LY%) (%)', value=25.0, format="%.1f")

# Create prediction button
if st.button('Predict'):
    # Prepare input data
    input_data = pd.DataFrame([[cl, glu, dbil, ldh, bun_scr, che, ibil, ua, pdw, ggt, ly_pct]], 
                              columns=['Cl', 'GLU', 'DBil', 'LDH', 'BUN/SCr', 'CHE', 'IBil', 'UA', 'PDW', 'GGT', 'LY%'])

    # Make prediction
    prediction = model.predict_proba(input_data)[0]
    
    # Display prediction results
    st.write("---")
    st.subheader("Prediction Results")
    
    # Display risk using metric
    st.metric(
        label="Risk of Severe Pneumonia",
        value=f"{prediction[1]:.1%}",
        delta=None
    )
    
    # Display risk level
    risk_level = "High Risk" if prediction[1] > 0.5 else "Low Risk"
    st.info(f"Risk Level: {risk_level}")

    # SHAP value explanation
    st.write("---")
    st.subheader("Model Interpretation")
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    # Plot force plot
    plt.figure(figsize=(15, 6))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_data.iloc[0],
        matplotlib=True,
        show=False,
        text_rotation=0
    )
    plt.gcf().set_facecolor('white')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)  
    st.pyplot(plt)
    plt.close()

    # Display feature importance explanation
    st.write("---")
    st.subheader("Feature Contribution Analysis")
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': input_data.columns,
        'SHAP Value': np.abs(shap_values[0])
    }).sort_values('SHAP Value', ascending=False)
    
    # Display feature importance table
    st.table(feature_importance)

# Add instructions
st.write("---")
st.markdown("""
### Instructions:
1. Enter the patient's clinical indicators in the input fields above
2. Click the "Predict" button to get results
3. The system will display the risk of severe pneumonia and risk level
4. SHAP values show how each feature contributes to the prediction
""")

# Add model information
st.sidebar.title("Model Information")
st.sidebar.info("""
- Model Type: CatBoost Classifier
- Training Data: Clinical Data
- Target Variable: Severe Pneumonia
- Number of Features: 11 Clinical Indicators
""")

# Add feature description
st.sidebar.title("Feature Description")
st.sidebar.markdown("""
- Cl: Chloride (mmol/L)
- GLU: Glucose (mmol/L)
- DBil: Direct Bilirubin (μmol/L)
- LDH: Lactate Dehydrogenase (U/L)
- BUN/SCr: Blood Urea Nitrogen/Serum Creatinine Ratio
- CHE: Cholinesterase (U/L)
- IBil: Indirect Bilirubin (μmol/L)
- UA: Uric Acid (μmol/L)
- PDW: Platelet Distribution Width (%)
- GGT: Gamma-Glutamyl Transferase (U/L)
- LY%: Lymphocyte Percentage (%)
""")