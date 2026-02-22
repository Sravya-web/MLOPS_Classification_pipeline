"""Streamlit frontend for ML classification predictions."""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="ML Classification Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Configuration ====================

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = 10

# ==================== Sidebar ====================

with st.sidebar:
    st.image("https://via.placeholder.com/200x60/4CAF50/FFFFFF?text=ML+System", use_column_width=True)
    st.title("Settings")
    
    api_url = st.text_input("API URL", value=API_URL)
    
    st.divider()
    st.subheader("📊 Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Home", "🔮 Single Prediction", "📦 Batch Prediction", "📈 Analytics", "❓ About"]
    )

# ==================== Helper Functions ====================

def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{api_url}/health", timeout=API_TIMEOUT)
        return response.status_code == 200
    except:
        return False


def get_prediction(feature1, feature2, feature3, feature4, feature5):
    """Get prediction from API."""
    try:
        payload = {
            "feature1": feature1,
            "feature2": feature2,
            "feature3": feature3,
            "feature4": feature4,
            "feature5": feature5
        }
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def get_batch_predictions(features_list):
    """Get batch predictions from API."""
    try:
        payload = [
            {
                "feature1": f[0],
                "feature2": f[1],
                "feature3": f[2],
                "feature4": f[3],
                "feature5": f[4]
            }
            for f in features_list
        ]
        response = requests.post(
            f"{api_url}/predict-batch",
            json=payload,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def get_model_info():
    """Get model information from API."""
    try:
        response = requests.get(f"{api_url}/info", timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except:
        return None


# ==================== Home Page ====================

if page == "🏠 Home":
    st.title("🤖 ML Classification System")
    
    # API Status
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("API Status")
        if check_api_health():
            st.success("✅ API is Running")
        else:
            st.error(f"❌ Cannot connect to API at {api_url}")
    
    with col2:
        st.subheader("Model Information")
        model_info = get_model_info()
        if model_info:
            st.json(model_info)
    
    st.divider()
    
    # Overview
    st.subheader("📋 System Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", "Random Forest")
    with col2:
        st.metric("Classes", "2 (Binary)")
    with col3:
        st.metric("Features", "5")
    with col4:
        st.metric("API Version", "1.0.0")
    
    st.divider()
    
    # Features explanation
    st.subheader("📊 Input Features")
    features_df = pd.DataFrame({
        "Feature": ["feature1", "feature2", "feature3", "feature4", "feature5"],
        "Type": ["Float", "Float", "Float", "Float", "Float"],
        "Description": ["Numeric feature 1", "Numeric feature 2", "Numeric feature 3", "Numeric feature 4", "Numeric feature 5"]
    })
    st.dataframe(features_df, use_container_width=True)


# ==================== Single Prediction ====================

elif page == "🔮 Single Prediction":
    st.title("🔮 Make a Single Prediction")
    
    # Check API
    if not check_api_health():
        st.error(f"❌ API is not available at {api_url}")
        st.stop()
    
    st.write("Enter feature values below to get a prediction:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature1 = st.number_input("Feature 1", value=1.2, step=0.1, format="%.2f")
        feature2 = st.number_input("Feature 2", value=3.4, step=0.1, format="%.2f")
        feature3 = st.number_input("Feature 3", value=5.6, step=0.1, format="%.2f")
    
    with col2:
        feature4 = st.number_input("Feature 4", value=2.1, step=0.1, format="%.2f")
        feature5 = st.number_input("Feature 5", value=1.5, step=0.1, format="%.2f")
    
    st.divider()
    
    # Prediction button
    if st.button("🚀 Get Prediction", use_container_width=True, type="primary"):
        with st.spinner("Getting prediction..."):
            result = get_prediction(feature1, feature2, feature3, feature4, feature5)
        
        if result:
            st.success("✅ Prediction Received!")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📌 Prediction Result")
                st.metric("Predicted Class", result["prediction"])
                st.metric("Confidence", f"{result['confidence']*100:.2f}%")
            
            with col2:
                st.subheader("📊 Class Probabilities")
                fig = go.Figure(data=[
                    go.Bar(
                        x=["Class 0", "Class 1"],
                        y=[result["probability_class_0"], result["probability_class_1"]],
                        marker_color=["#FF6B6B", "#4ECDC4"]
                    )
                ])
                fig.update_layout(
                    title="Prediction Probabilities",
                    yaxis_title="Probability",
                    xaxis_title="Class",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Input summary
            st.divider()
            st.subheader("📥 Input Summary")
            input_df = pd.DataFrame({
                "Feature": ["feature1", "feature2", "feature3", "feature4", "feature5"],
                "Value": [feature1, feature2, feature3, feature4, feature5]
            })
            st.dataframe(input_df, use_container_width=True)
            
            # Timestamp
            st.caption(f"Prediction made at: {result['timestamp']}")


# ==================== Batch Prediction ====================

elif page == "📦 Batch Prediction":
    st.title("📦 Batch Predictions")
    
    # Check API
    if not check_api_health():
        st.error(f"❌ API is not available at {api_url}")
        st.stop()
    
    st.write("Upload a CSV file with features or paste data below:")
    
    tab1, tab2 = st.tabs(["CSV Upload", "Manual Input"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded data:")
            st.dataframe(df, use_container_width=True)
            
            if st.button("🚀 Get Batch Predictions", use_container_width=True, type="primary"):
                with st.spinner("Processing batch predictions..."):
                    # Extract features
                    feature_cols = [col for col in df.columns if col.startswith("feature")]
                    if len(feature_cols) < 5:
                        st.error("CSV must contain at least 5 feature columns")
                    else:
                        features_list = df[["feature1", "feature2", "feature3", "feature4", "feature5"]].values.tolist()
                        predictions = get_batch_predictions(features_list)
                
                if predictions:
                    st.success(f"✅ Got predictions for {len(predictions)} samples")
                    
                    # Add predictions to dataframe
                    results_df = df.copy()
                    results_df["Prediction"] = [p["prediction"] for p in predictions]
                    results_df["Confidence"] = [p["confidence"] for p in predictions]
                    results_df["Prob_Class_0"] = [p["probability_class_0"] for p in predictions]
                    results_df["Prob_Class_1"] = [p["probability_class_1"] for p in predictions]
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    with tab2:
        st.write("Manually enter sample data:")
        
        num_samples = st.number_input("Number of samples", min_value=1, max_value=100, value=5)
        
        manual_data = []
        for i in range(num_samples):
            cols = st.columns(5)
            with cols[0]:
                f1 = st.number_input(f"F1 ({i+1})", value=1.2, step=0.1, key=f"f1_{i}")
            with cols[1]:
                f2 = st.number_input(f"F2 ({i+1})", value=3.4, step=0.1, key=f"f2_{i}")
            with cols[2]:
                f3 = st.number_input(f"F3 ({i+1})", value=5.6, step=0.1, key=f"f3_{i}")
            with cols[3]:
                f4 = st.number_input(f"F4 ({i+1})", value=2.1, step=0.1, key=f"f4_{i}")
            with cols[4]:
                f5 = st.number_input(f"F5 ({i+1})", value=1.5, step=0.1, key=f"f5_{i}")
            
            manual_data.append([f1, f2, f3, f4, f5])
        
        if st.button("🚀 Predict", use_container_width=True, type="primary"):
            with st.spinner("Getting predictions..."):
                predictions = get_batch_predictions(manual_data)
            
            if predictions:
                st.success(f"✅ Got predictions for {len(predictions)} samples")
                
                results_df = pd.DataFrame({
                    "Feature1": [d[0] for d in manual_data],
                    "Feature2": [d[1] for d in manual_data],
                    "Feature3": [d[2] for d in manual_data],
                    "Feature4": [d[3] for d in manual_data],
                    "Feature5": [d[4] for d in manual_data],
                    "Prediction": [p["prediction"] for p in predictions],
                    "Confidence": [f"{p['confidence']*100:.2f}%" for p in predictions]
                })
                
                st.dataframe(results_df, use_container_width=True)


# ==================== Analytics ====================

elif page == "📈 Analytics":
    st.title("📈 Model Analytics")
    
    st.info("📊 Analytics and monitoring dashboard would be displayed here")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Predictions (Today)", "0")
    with col2:
        st.metric("Avg Confidence", "0%")
    with col3:
        st.metric("API Uptime", "100%")
    
    st.divider()
    
    # Sample chart
    st.subheader("Prediction Distribution")
    sample_data = {
        "Hour": [f"0{i}:00" for i in range(1, 9)],
        "Predictions": [10, 15, 12, 18, 16, 20, 14, 19]
    }
    sample_df = pd.DataFrame(sample_data)
    
    fig = go.Figure(data=[
        go.Bar(x=sample_df["Hour"], y=sample_df["Predictions"])
    ])
    fig.update_layout(title="Predictions by Hour", xaxis_title="Time", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)


# ==================== About ====================

elif page == "❓ About":
    st.title("❓ About This System")
    
    st.subheader("🎯 Purpose")
    st.write("""
    This is a complete ML classification system demonstrating MLOps best practices:
    - **Data Management**: PostgreSQL database integration
    - **Model Training**: Scikit-learn pipelines with hyperparameter tuning
    - **Experiment Tracking**: Weights & Biases integration
    - **API**: FastAPI with monitoring and logging
    - **Frontend**: Streamlit interface
    - **Deployment**: Docker containerization and cloud deployment
    """)
    
    st.subheader("🔧 Technical Stack")
    tech_stack = {
        "Component": ["Backend", "Frontend", "Database", "ML", "Monitoring", "Deployment"],
        "Technology": ["FastAPI", "Streamlit", "PostgreSQL", "Scikit-learn", "Prometheus/Grafana", "Docker"]
    }
    st.table(tech_stack)
    
    st.subheader("📚 Documentation")
    st.markdown("""
    - [API Docs](http://localhost:8000/docs)
    - [GitHub Repository](#)
    - [README](#)
    """)
    
    st.subheader("👨‍💻 Developer Info")
    st.write("Built with ❤️ for MLOps excellence")

st.divider()
st.caption("ML Classification System v1.0.0 | " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
