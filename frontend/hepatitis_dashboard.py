"""Hepatitis Classification Dashboard - Specialized Analytics."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Hepatitis Classification Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .header-title {
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .alert-box {
        background-color: #FFE66D;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #FF6B6B;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
API_URL = "http://localhost:8000"
MODEL_PATH = "model_artifacts/hepatitis_model.joblib"
PREPROCESSOR_PATH = "model_artifacts/hepatitis_preprocessor.joblib"
DATA_PATH = "data/hepatitis.csv"

# ==================== Helper Functions ====================

@st.cache_data(ttl=300)
def load_hepatitis_data():
    """Load hepatitis dataset."""
    try:
        df = pd.read_csv(DATA_PATH, index_col=0)
        df = df.dropna(subset=['Category'])
        return df
    except:
        return None


@st.cache_data(ttl=300)
def load_model_and_preprocessor():
    """Load trained model and preprocessor."""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except:
        return None, None


def prepare_hepatitis_features(features_dict):
    """Prepare features for prediction.
    
    Args:
        features_dict: Dictionary with feature values
        
    Returns:
        Numpy array of features
    """
    feature_names = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
    values = []
    for name in feature_names:
        if name == 'Sex':
            # Convert m/f to 0/1
            value = 0 if features_dict.get(name) == 'm' else 1
        else:
            value = features_dict.get(name, 0)
        values.append(value)
    return np.array(values).reshape(1, -1)


def get_hepatitis_category_color(category_idx):
    """Get color for hepatitis category."""
    colors = {
        0: "#2ecc71",  # Green - Blood Donor
        1: "#e74c3c",  # Red - Hepatitis
        2: "#f39c12",  # Orange - Fibrosis
        3: "#8b0000",  # Dark Red - Cirrhosis
        4: "#95a5a6"   # Gray - Suspect
    }
    return colors.get(category_idx, "#3498db")


def generate_hepatitis_report(df, prediction_result):
    """Generate hepatitis analysis report."""
    report = f"""
    ### 📋 Hepatitis Classification Report
    
    **Patient Profile:**
    - Age: {prediction_result.get('age', 'N/A')} years
    - Sex: {prediction_result.get('sex', 'N/A')}
    
    **Prediction Result:** {prediction_result.get('category', 'Unknown')}
    - Confidence: {prediction_result.get('confidence', 0):.2%}
    
    **Lab Values:**
    - ALB (Albumin): {prediction_result.get('ALB', 'N/A')}
    - ALP (Alkaline Phosphatase): {prediction_result.get('ALP', 'N/A')}
    - ALT (Alanine Aminotransferase): {prediction_result.get('ALT', 'N/A')}
    - AST (Aspartate Aminotransferase): {prediction_result.get('AST', 'N/A')}
    - BIL (Bilirubin): {prediction_result.get('BIL', 'N/A')}
    - GGT (Gamma-Glutamyl Transferase): {prediction_result.get('GGT', 'N/A')}
    """
    return report


# ==================== Sidebar ====================

with st.sidebar:
    st.image("https://via.placeholder.com/200x60/FF6B6B/FFFFFF?text=Hepatitis+Classifier", use_container_width=True)
    
    st.title("Navigation")
    
    dashboard_type = st.radio(
        "Select Dashboard",
        ["Overview", "Patient Analysis", "Feature Analysis", "Statistics", "Clinical Insights"]
    )
    
    st.divider()
    
    # Dataset Statistics
    df = load_hepatitis_data()
    if df is not None:
        st.subheader("Dataset Info")
        st.metric("Total Patients", len(df))
        
        category_counts = df['Category'].value_counts()
        for category, count in category_counts.items():
            st.text(f"{category}: {count}")


# ==================== Main Content ====================

st.markdown("<h1 class='header-title'>Hepatitis Classification Dashboard</h1>", unsafe_allow_html=True)

# Load data
df = load_hepatitis_data()
model, preprocessor = load_model_and_preprocessor()

# ==================== OVERVIEW PAGE ====================

if dashboard_type == "Overview":
    st.write("Complete hepatitis classification system overview with dataset analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", len(df) if df is not None else 0)
    
    with col2:
        healthy = len(df[df['Category'] == '0=Blood Donor']) if df is not None else 0
        st.metric("Blood Donors", healthy)
    
    with col3:
        hepatitis = len(df[df['Category'] == '1=Hepatitis']) if df is not None else 0
        st.metric("Hepatitis Cases", hepatitis)
    
    with col4:
        cirrhosis = len(df[df['Category'] == '3=Cirrhosis']) if df is not None else 0
        st.metric("Cirrhosis Cases", cirrhosis)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    # Disease Distribution
    with col1:
        if df is not None:
            category_counts = df['Category'].value_counts()
            fig = go.Figure(data=[
                go.Pie(
                    labels=category_counts.index,
                    values=category_counts.values,
                    marker=dict(colors=['#2ecc71', '#e74c3c', '#f39c12', '#8b0000', '#95a5a6'][:len(category_counts)])
                )
            ])
            fig.update_layout(title="Patient Distribution by Category", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Age Distribution
    with col2:
        if df is not None:
            fig = go.Figure()
            for category in df['Category'].unique():
                category_data = df[df['Category'] == category]['Age']
                fig.add_trace(go.Box(y=category_data, name=category))
            fig.update_layout(title="Age Distribution by Category", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Key Statistics
    st.subheader("Key Statistics")
    if df is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Age", f"{df['Age'].mean():.1f} years")
        
        with col2:
            male_pct = (df['Sex'] == 'm').sum() / len(df) * 100
            st.metric("Male Patients", f"{male_pct:.1f}%")
        
        with col3:
            st.metric("Data Records", len(df))


# ==================== PATIENT ANALYSIS ====================

elif dashboard_type == "Patient Analysis":
    st.write("Analyze individual patient lab values and get hepatitis classification predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=45)
        sex = st.selectbox("Sex", ["m", "f"])
    
    with col2:
        st.subheader("Lab Values")
        alb = st.number_input("ALB (Albumin)", min_value=10.0, max_value=85.0, value=40.0)
        alp = st.number_input("ALP (Alkaline Phosphatase)", min_value=10.0, max_value=450.0, value=70.0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alt = st.number_input("ALT (Alanine Aminotransferase)", min_value=0.0, max_value=200.0, value=30.0)
        ast = st.number_input("AST (Aspartate Aminotransferase)", min_value=0.0, max_value=200.0, value=35.0)
    
    with col2:
        bil = st.number_input("BIL (Bilirubin)", min_value=0.0, max_value=60.0, value=10.0)
        che = st.number_input("CHE (Cholinesterase)", min_value=1.0, max_value=15.0, value=7.0)
    
    with col3:
        chol = st.number_input("CHOL (Cholesterol)", min_value=2.0, max_value=10.0, value=5.0)
        crea = st.number_input("CREA (Creatinine)", min_value=5.0, max_value=200.0, value=80.0)
        ggt = st.number_input("GGT (Gamma-Glutamyl)", min_value=0.0, max_value=700.0, value=30.0)
        prot = st.number_input("PROT (Protein)", min_value=40.0, max_value=95.0, value=72.0)
    
    st.divider()
    
    # Prediction
    if st.button("Analyze Patient", use_container_width=True, type="primary"):
        if model is not None and preprocessor is not None:
            features_dict = {
                'Age': age, 'Sex': sex, 'ALB': alb, 'ALP': alp, 'ALT': alt, 'AST': ast,
                'BIL': bil, 'CHE': che, 'CHOL': chol, 'CREA': crea, 'GGT': ggt, 'PROT': prot
            }
            
            X = prepare_hepatitis_features(features_dict)
            X_scaled = preprocessor.transform(X)
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            
            # Map prediction
            category_map = {0: "Blood Donor", 1: "Hepatitis", 2: "Fibrosis", 3: "Cirrhosis", 4: "Suspect"}
            predicted_category = category_map.get(prediction, "Unknown")
            confidence = probabilities[prediction]
            
            # Display result
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                    <div class='alert-box'>
                    <h3>Prediction Result</h3>
                    <h2>{predicted_category}</h2>
                    <p>Confidence: <strong>{confidence:.2%}</strong></p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Probability bar chart
                prob_df = pd.DataFrame({
                    'Category': [category_map.get(i, "Unknown") for i in range(len(probabilities))],
                    'Probability': probabilities
                })
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=prob_df['Probability'],
                        y=prob_df['Category'],
                        orientation='h',
                        marker=dict(color=prob_df['Probability'], colorscale='RdYlGn')
                    )
                ])
                fig.update_layout(title="Classification Probabilities", xaxis_title="Probability", height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("⚠️ Model not loaded. Please train the model first using: `python scripts/train_hepatitis.py`")


# ==================== FEATURE ANALYSIS ====================

elif dashboard_type == "Feature Analysis":
    st.write("Analyze laboratory features and their distributions by hepatitis category")
    
    if df is not None:
        # Select feature to analyze
        numeric_features = ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
        selected_feature = st.selectbox("Select Feature to Analyze", numeric_features)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        # Distribution by category
        with col1:
            fig = go.Figure()
            for category in sorted(df['Category'].unique()):
                feature_data = df[df['Category'] == category][selected_feature].dropna()
                fig.add_trace(go.Box(y=feature_data, name=category))
            fig.update_layout(title=f"{selected_feature} Distribution by Category", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Histogram
        with col2:
            fig = go.Figure()
            for category in sorted(df['Category'].unique()):
                feature_data = df[df['Category'] == category][selected_feature].dropna()
                fig.add_trace(go.Histogram(x=feature_data, name=category, opacity=0.7))
            fig.update_layout(title=f"{selected_feature} Histogram", barmode='overlay', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Statistics table
        st.subheader("Feature Statistics by Category")
        stats_data = []
        for category in sorted(df['Category'].unique()):
            feature_data = df[df['Category'] == category][selected_feature].dropna()
            stats_data.append({
                'Category': category,
                'Count': len(feature_data),
                'Mean': f"{feature_data.mean():.2f}",
                'Median': f"{feature_data.median():.2f}",
                'Std Dev': f"{feature_data.std():.2f}",
                'Min': f"{feature_data.min():.2f}",
                'Max': f"{feature_data.max():.2f}"
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)


# ==================== STATISTICS ====================

elif dashboard_type == "Statistics":
    st.write("Overall dataset statistics and correlations")
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        # Correlation heatmap
        with col1:
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            corr_matrix = numeric_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu'
            ))
            fig.update_layout(title="Feature Correlation Matrix", height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Missing values
        with col2:
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig = go.Figure(data=[
                    go.Bar(x=missing_data.values, y=missing_data.index, orientation='h')
                ])
                fig.update_layout(title="Missing Values Count", xaxis_title="Count", height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("✅ No missing values in dataset")


# ==================== CLINICAL INSIGHTS ====================

elif dashboard_type == "Clinical Insights":
    st.write("Clinical insights and hepatitis classification patterns")
    
    st.markdown("""
    ### 🏥 Hepatitis Classification Categories
    
    **0 = Blood Donor** 🟢
    - Healthy control group
    - Normal liver function
    - No hepatitis markers
    
    **1 = Hepatitis** 🔴
    - Active hepatitis infection
    - Elevated liver enzymes (ALT, AST)
    - Abnormal bilirubin levels
    
    **2 = Fibrosis** 🟠
    - Liver tissue scarring
    - Intermediate stage of disease
    - Requires monitoring
    
    **3 = Cirrhosis** 🔴🔴
    - Advanced liver disease
    - Severe fibrosis
    - Significantly altered liver function
    
    **0s = Suspect Blood Donor** ⚫
    - Borderline cases
    - Requires further testing
    """)
    
    st.divider()
    
    if df is not None:
        st.subheader("📊 Key Lab Markers by Category")
        
        categories = sorted(df['Category'].unique())
        selected_category = st.selectbox("Select Category", categories)
        
        category_data = df[df['Category'] == selected_category]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg ALT (AST)", f"{category_data['ALT'].mean():.1f}")
        
        with col2:
            st.metric("Avg AST", f"{category_data['AST'].mean():.1f}")
        
        with col3:
            st.metric("Avg Bilirubin", f"{category_data['BIL'].mean():.1f}")
        
        st.divider()
        
        st.subheader("🔬 Clinical Recommendations")
        
        if "Cirrhosis" in selected_category:
            st.warning("""
            **⚠️ Advanced Disease Stage**
            - Regular monitoring required
            - Hepatologist consultation recommended
            - Screen for hepatocellular carcinoma
            - Evaluate for transplant eligibility
            """)
        
        elif "Hepatitis" in selected_category:
            st.warning("""
            **⚠️ Active Hepatitis**
            - Antiviral therapy consideration
            - Monitor viral load
            - Assess fibrosis progression
            - Lifestyle modifications needed
            """)
        
        elif "Fibrosis" in selected_category:
            st.info("""
            **ℹ️ Liver Fibrosis**
            - Monitor disease progression
            - Treat underlying cause
            - Reduce alcohol consumption
            - Regular follow-up ultrasound
            """)
        
        else:
            st.success("""
            **✅ Normal Liver Function**
            - Continue regular check-ups
            - Maintain healthy lifestyle
            - Vaccinate if needed
            - Avoid hepatotoxic substances
            """)


# ==================== Footer ====================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🏥 Hepatitis Classification System")

with col2:
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col3:
    st.caption("v1.0.0 | Clinical Decision Support Tool")
