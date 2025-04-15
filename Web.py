import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
import matplotlib.pyplot as plt

# 设置页面标题和全局字体大小
st.set_page_config(page_title="Crohn's Disease Prediction Model", layout="wide")

# 增大全局字体大小的CSS
st.markdown("""
<style>
    html, body, [class*="st-"] {
        font-size: 18px !important;
    }
    .stSelectbox label, .stNumberInput label {
        font-size: 20px !important;
    }
    .stButton button {
        font-size: 20px !important;
    }
    .stMarkdown h1, h2, h3 {
        font-size: 28px !important;
    }
    .stTitle {
        font-size: 32px !important;
    }
    .stSubheader {
        font-size: 24px !important;
    }
    .stSidebar .stMarkdown {
        font-size: 18px !important;
    }
</style>
""", unsafe_allow_html=True)

# 加载模型
@st.cache_resource
def load_model():
    model = load('GBM_model.pkl')
    return model

model = load_model()

# 创建页面标题
st.title("Crohn's Disease Prediction Model")
st.write("Please input patient's clinical indicators:")

# 创建输入字段
col1, col2 = st.columns(2)

# 在创建输入字段部分修改变量名
with col1:
    abdominal_pain = st.selectbox('Abdominal pain lasting for at least one month', ['No', 'Yes'])
    perianal_disease = st.selectbox('Perianal disease', ['No', 'Yes'])  # 修改变量名
    weight_loss = st.selectbox('Weight loss', ['No', 'Yes'])
    Age = st.number_input('Age (years)', value=30, format="%d")
    bmi = st.number_input('BMI (kg/m²)', value=22.0, format="%.1f")

with col2:
    hb = st.number_input('Hemoglobin (Hb) (g/L)', value=120.0, format="%.1f")
    plt_count = st.number_input('Platelet count (PLT) (10^9/L)', value=200.0, format="%.1f")
    mcv = st.number_input('Mean corpuscular volume (MCV) (fL)', value=85.0, format="%.1f")
    lym = st.number_input('Lymphocyte cell count (LYM) (10^9/L)', value=1.5, format="%.2f")
    alb = st.number_input('Albumin (ALB) (g/L)', value=40.0, format="%.1f")

# 创建预测按钮
# 在准备输入数据部分修改变量名
if st.button('Predict'):
    # 准备输入数据
    input_data = pd.DataFrame({
        'Abdominal_pain': [abdominal_pain],
        'Perianal_disease': [perianal_disease],  # 修改变量名
        'Weight_loss': [weight_loss],
        'Age': [Age],
        'BMI': [bmi],
        'Hb': [hb],
        'PLT': [plt_count],
        'MCV': [mcv],
        'LYM': [lym],
        'ALB': [alb]
    })
    
    categorical_features = ['Abdominal_pain', 'Perianal_disease', 'Weight_loss']  # 修改变量名
    input_data_model = input_data.copy()
    
    for col in categorical_features:
        input_data_model[col] = input_data_model[col].map({'Yes': 1, 'No': 0})
    
    expected_columns = ['Abdominal_pain', 'Perianal_disease', 'Weight_loss', 'Age', 'BMI', 'Hb', 'PLT', 'MCV', 'LYM', 'ALB']  # 修改变量名
    input_data_model = input_data_model[expected_columns]
    
    # 进行预测
    prediction = model.predict_proba(input_data_model)[0]
    
    # 显示预测结果
    st.write("---")
    st.subheader("Prediction Results")
    
    # 使用进度条显示疾病风险
    st.metric(
        label="Crohn's Disease Risk",
        value=f"{prediction[1]:.1%}",
        delta=None
    )
    
    # 显示风险等级
    risk_level = "High Risk" if prediction[1] > 0.5 else "Low Risk"
    st.info(f"Risk Level: {risk_level}")

    # SHAP值解释
    st.write("---")
    st.subheader("Model Interpretation")
    
    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data_model)
    
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_plot = shap_values[1]
        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    else:
        shap_values_plot = shap_values
        expected_value = explainer.expected_value
    
    # 使用瀑布图代替力图
    st.write("SHAP Waterfall Plot (showing how each feature contributes to the prediction):")
    
    # 创建SHAP解释对象
    shap_explanation = shap.Explanation(
        values=shap_values_plot[0], 
        base_values=expected_value, 
        data=input_data_model.iloc[0].values,
        feature_names=list(input_data_model.columns)
    )
    
    # 绘制瀑布图 
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_explanation, max_display=10, show=False)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()
    
    # 决策图
    st.write("SHAP Decision Plot:")
    plt.figure(figsize=(8, 6))
    shap.decision_plot(expected_value, shap_values_plot[0], input_data_model.iloc[0], 
                      feature_names=list(input_data_model.columns), show=False)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()
    
    # 显示特征重要性说明
    st.write("---")
    st.subheader("Feature Contribution Analysis")
    
    # 获取特征重要性 - 修复SHAP值处理

    if isinstance(shap_values, list) and len(shap_values) > 1:
        # For binary classification, use class 1 (positive class)
        shap_values_processed = shap_values[1][0]
    else:
        # For single array of SHAP values
        shap_values_processed = shap_values[0]
    
    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'Feature': input_data_model.columns,
        'SHAP Value': np.abs(shap_values_processed)
    }).sort_values('SHAP Value', ascending=False)
    
    # 将特征名称映射为更易读的形式
    feature_mapping = {
        'Age': 'Age (years)',
        'BMI': 'BMI (kg/m²)',
        'Hb': 'Hemoglobin (Hb) (g/L)',
        'PLT': 'Platelet count (PLT) (10^9/L)',
        'MCV': 'Mean corpuscular volume (MCV) (fL)',
        'LYM': 'Lymphocyte cell count (LYM) (10^9/L)',
        'ALB': 'Albumin (ALB) (g/L)',
        'Abdominal_pain': 'Abdominal pain lasting for at least one month',
        'Perianal_diseases': 'Perianal disease',
        'Weight_loss': 'Weight loss'
    }
    

    feature_importance['Feature'] = feature_importance['Feature'].map(feature_mapping)
    
    # 显示特征重要性表格
    st.table(feature_importance)

# 添加说明信息
st.write("---")
st.markdown("""
### Instructions:
1. Enter the patient's clinical indicators in the input fields above
2. Click the "Predict" button to get results
3. The system will display the Crohn's Disease risk and risk level
4. SHAP values show how each feature contributes to the prediction
""")

# 添加模型信息
st.sidebar.title("Model Information")
st.sidebar.info("""
- Model Type: Gradient Boosting Classifier
- Training Data: Clinical Patient Data
- Target Variable: Crohn's Disease Diagnosis
- Number of Features: 10 Clinical Indicators
""")

# 添加特征说明
st.sidebar.title("Feature Description")
st.sidebar.markdown("""
- **Abdominal pain lasting for at least one month**: Persistent abdominal pain for more than one month
- **Perianal diseases**: Presence of perianal disease
- **Weight loss**: Significant weight loss
- **Age**: Patient's age in years
- **BMI**: Body Mass Index (kg/m²)
- **Hemoglobin (Hb)**: Hemoglobin level in g/L
- **Platelet count (PLT)**: Platelet count in 10^9/L
- **Mean corpuscular volume (MCV)**: Average red blood cell volume in fL
- **Lymphocyte count (LYM)**: Lymphocyte cell count in 10^9/L
- **Albumin (ALB)**: Albumin level in g/L
""")

# 添加关于页面
st.sidebar.title("About")
st.sidebar.info("""
This application predicts the risk of Crohn's Disease based on clinical indicators. 
The model was trained on patient data and uses a Gradient Boosting algorithm to make predictions.
""")