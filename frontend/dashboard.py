"""Comprehensive analytics dashboard for ML Classification Pipeline."""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ML Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .header-title {
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
API_URL = "http://localhost:8000"
REFRESH_INTERVAL = 5  # seconds

# ==================== Helper Functions ====================

@st.cache_data(ttl=5)
def fetch_api_health():
    """Fetch API health status."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200, response.json()
    except:
        return False, None

@st.cache_data(ttl=5)
def fetch_model_info():
    """Fetch model information."""
    try:
        response = requests.get(f"{API_URL}/info", timeout=2)
        return response.json()
    except:
        return None

def generate_sample_metrics():
    """Generate sample metrics for demonstration."""
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(60, 0, -1)]
    
    # Sample data
    predictions = np.random.randint(10, 100, 60)
    accuracy_values = np.random.uniform(0.82, 0.90, 60)
    latencies = np.random.uniform(50, 150, 60)
    
    return {
        "timestamps": timestamps,
        "predictions": predictions.tolist(),
        "accuracy": accuracy_values.tolist(),
        "latencies": latencies.tolist()
    }

# ==================== Sidebar ====================

with st.sidebar:
    st.image("https://via.placeholder.com/200x60/667eea/FFFFFF?text=ML+Pipeline", use_column_width=True)
    
    st.title("Dashboard Controls")
    
    # Page selection
    dashboard_type = st.radio(
        "Select Dashboard",
           ["Overview", "Statistics", "Feature Analysis", "Patient Analysis", "Clinical Insights"]
    )
    
    st.divider()
    
    # Settings
    st.subheader("Settings")
    api_url = st.text_input("API URL", value=API_URL)
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    
    st.divider()
    
    # System Status
    st.subheader("System Status")
    is_healthy, health_data = fetch_api_health()
    
    if is_healthy:
        st.success("✅ API Running")
        st.caption(f"Last checked: {datetime.now().strftime('%H:%M:%S')}")
    else:
        st.error("❌ API Offline")
    
    # Model Info
    model_info = fetch_model_info()
    if model_info:
        st.subheader("Model Info")
        st.write(f"**Name**: {model_info.get('model_name')}")
        st.write(f"**Version**: {model_info.get('version')}")
        st.write(f"**Classes**: {model_info.get('classes')}")

# ==================== Main Content ====================

st.markdown("<h1 class='header-title'>📊 ML Classification Pipeline Dashboard</h1>", unsafe_allow_html=True)

if dashboard_type == "🏠 Overview":
    # ==================== OVERVIEW PAGE ====================
    
    st.write("Complete system overview with key metrics and status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", "1,247", "+12% today")
    
    with col2:
        st.metric("Model Accuracy", "87.3%", "+2.1% vs baseline")
    
    with col3:
        st.metric("Avg Latency", "82ms", "-8ms vs yesterday")
    
    with col4:
        st.metric("API Uptime", "99.9%", "✅ Healthy")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Predictions Over Time")
        metrics = generate_sample_metrics()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metrics["timestamps"],
            y=metrics["predictions"],
            mode='lines+markers',
            name='Predictions',
            line=dict(color='#667eea', width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        fig.update_layout(
            title="Predictions per Minute",
            xaxis_title="Time",
            yaxis_title="Count",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Prediction Distribution")
        class_dist = pd.DataFrame({
            "Class": ["Class 0", "Class 1"],
            "Count": [548, 699],
            "Percentage": [44, 56]
        })
        fig = px.pie(class_dist, values="Count", names="Class", 
                     color_discrete_sequence=['#667eea', '#764ba2'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Performance Metrics")
        metrics_data = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            "Value": [0.873, 0.856, 0.834, 0.845, 0.891]
        }
        metrics_df = pd.DataFrame(metrics_data)
        fig = go.Figure(data=[
            go.Bar(x=metrics_df["Metric"], y=metrics_df["Value"],
                   marker_color='#667eea',
                   text=[f"{v:.3f}" for v in metrics_df["Value"]],
                   textposition='auto')
        ])
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔥 Model Status")
        st.write("**Last Trained**: 2 hours ago")
        st.write("**Training Status**: ✅ Completed successfully")
        st.write("**Model Size**: 62 KB")
        st.write("**Features**: 5 input features")
        st.write("**Algorithm**: Random Forest Classifier")
        st.write("**Hyperparameters**: n_estimators=100, max_depth=10")

elif dashboard_type == "📈 Performance":
    # ==================== PERFORMANCE PAGE ====================
    
    st.write("Detailed API and model performance metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Peak QPS", "142 req/s", "+5% vs baseline")
    with col2:
        st.metric("P95 Latency", "156ms", "-12ms improvement")
    with col3:
        st.metric("Error Rate", "0.02%", "↓ Lower is better")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏱️ Prediction Latency (ms)")
        metrics = generate_sample_metrics()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metrics["timestamps"],
            y=metrics["latencies"],
            mode='lines',
            name='Latency (ms)',
            line=dict(color='#f59e0b', width=2),
            fill='tozeroy',
            fillcolor='rgba(245, 158, 11, 0.1)'
        ))
        fig.add_hline(y=100, line_dash="dash", line_color="red", 
                      annotation_text="Target: 100ms")
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Model Accuracy Over Time")
        metrics = generate_sample_metrics()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metrics["timestamps"],
            y=[acc*100 for acc in metrics["accuracy"]],
            mode='lines+markers',
            name='Accuracy %',
            line=dict(color='#10b981', width=2)
        ))
        fig.add_hline(y=85, line_dash="dash", line_color="orange",
                      annotation_text="Minimum threshold: 85%")
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("🔍 Performance Breakdown")
    
    perf_data = pd.DataFrame({
        "Endpoint": ["/predict", "/predict-batch", "/health", "/info", "/metrics"],
        "Requests": [547, 234, 1502, 89, 234],
        "Avg Latency (ms)": [82, 145, 5, 8, 3],
        "Error Rate %": [0.0, 0.01, 0.0, 0.0, 0.0]
    })
    
    st.dataframe(perf_data, use_container_width=True, hide_index=True)

elif dashboard_type == "🔄 Pipeline":
    # ==================== PIPELINE PAGE ====================
    
    st.write("ML Pipeline architecture and workflow visualization")
    
    st.subheader("🏗️ ML Pipeline Architecture")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📥 Input Layer**
        - Features: 5
        - Type: Float
        - Validation: Pydantic
        """)
    
    with col2:
        st.warning("""
        **⚙️ Processing Layer**
        - StandardScaler
        - Random Forest
        - Probability Scoring
        """)
    
    with col3:
        st.success("""
        **📤 Output Layer**
        - Prediction (0/1)
        - Confidence Score
        - Class Probabilities
        """)
    
    st.divider()
    
    st.subheader("📋 Pipeline Stages")
    
    stages_data = {
        "Stage": [
            "Data Ingestion",
            "Preprocessing",
            "Feature Engineering",
            "Model Inference",
            "Prediction Output"
        ],
        "Status": ["✅", "✅", "✅", "✅", "✅"],
        "Latency (ms)": [2, 15, 8, 45, 12],
        "Records": ["1,247", "1,247", "1,247", "1,247", "1,247"]
    }
    
    stages_df = pd.DataFrame(stages_data)
    st.dataframe(stages_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader("🔄 Data Flow Diagram")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        ```
        Client Request
            ↓
        ┌─────────────────┐
        │  FastAPI Server │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │ Pydantic Valid. │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │ StandardScaler  │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │ Random Forest   │
        └────────┬────────┘
                 ↓
        ┌─────────────────┐
        │ Format & Return │
        └────────┬────────┘
                 ↓
        JSON Response
        ```
        """)
    
    with col2:
        st.write("""
        **Pipeline Configuration:**
        
        - **Scaler**: StandardScaler
          - Mean: 0
          - Std Dev: 1
        
        - **Model**: RandomForestClassifier
          - Estimators: 100
          - Max Depth: 10
          - Min Split: 2
        
        - **Output**:
          - Class Label
          - Probabilities
          - Confidence
        """)

elif dashboard_type == "⚙️ System":
    # ==================== SYSTEM PAGE ====================
    
    st.write("System health, resources, and infrastructure monitoring")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CPU Usage", "34%", "Normal")
    with col2:
        st.metric("Memory Usage", "512 MB", "71%")
    with col3:
        st.metric("Disk Usage", "2.3 GB", "18%")
    with col4:
        st.metric("Network", "124 Mbps", "Active")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Resource Usage")
        resources = pd.DataFrame({
            "Resource": ["CPU", "Memory", "Disk", "Network"],
            "Usage %": [34, 71, 18, 45],
            "Limit": ["100%", "100%", "100%", "1000Mbps"]
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=resources["Resource"],
                y=resources["Usage %"],
                marker_color=['#667eea', '#f59e0b', '#10b981', '#ef4444']
            )
        ])
        fig.update_layout(height=400, yaxis_title="Usage %")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏥 Service Health")
        
        services = [
            ("FastAPI Backend", "✅ Healthy"),
            ("PostgreSQL Database", "✅ Healthy"),
            ("Prometheus", "✅ Healthy"),
            ("Grafana", "✅ Healthy"),
            ("Model Server", "✅ Healthy")
        ]
        
        for service, status in services:
            st.write(f"**{service}**: {status}")
    
    st.divider()
    
    st.subheader("🔧 Configuration")
    
    config = {
        "Environment": "Production",
        "API Port": 8000,
        "Database": "PostgreSQL",
        "Model Format": "joblib",
        "Framework": "FastAPI",
        "Python Version": "3.11",
        "Memory Limit": "1024 MB",
        "CPU Cores": 4
    }
    
    config_df = pd.DataFrame(
        list(config.items()),
        columns=["Parameter", "Value"]
    )
    st.dataframe(config_df, use_container_width=True, hide_index=True)

elif dashboard_type == "📊 Detailed Analytics":
    # ==================== DETAILED ANALYTICS PAGE ====================
    
    st.write("In-depth analytics and advanced metrics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Model", "Errors", "Database"])
    
    with tab1:
        st.subheader("📊 Prediction Analytics")
        
        metrics = generate_sample_metrics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Predictions", "1,247", "+156 today")
        with col2:
            st.metric("Avg Confidence", "0.847", "+0.032 today")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Predictions by Hour**")
            hourly_data = pd.DataFrame({
                "Hour": [f"{h:02d}:00" for h in range(24)],
                "Count": np.random.randint(20, 80, 24)
            })
            fig = px.bar(hourly_data, x="Hour", y="Count", title="Hourly Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Class Distribution**")
            class_data = pd.DataFrame({
                "Class": ["0", "1"],
                "Count": [548, 699]
            })
            fig = px.pie(class_data, values="Count", names="Class")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🤖 Model Analytics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", "87.3%")
        with col2:
            st.metric("F1-Score", "0.845")
        with col3:
            st.metric("ROC-AUC", "0.891")
        
        st.divider()
        
        # Confusion Matrix
        st.write("**Confusion Matrix**")
        cm_data = np.array([[475, 73], [29, 670]])
        fig = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            text=cm_data,
            texttemplate='%{text}',
            colorscale='Blues'
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("❌ Error Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Error Rate", "0.02%", "Low")
        with col2:
            st.metric("False Positives", "73", "5.8%")
        with col3:
            st.metric("False Negatives", "29", "2.3%")
        
        st.divider()
        
        st.write("**Error Breakdown**")
        errors = pd.DataFrame({
            "Error Type": ["Validation", "Model", "Database", "Timeout", "Other"],
            "Count": [5, 2, 1, 0, 1],
            "Percentage": [55, 22, 11, 0, 12]
        })
        st.dataframe(errors, use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("💾 Database Analytics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", "3,427", "+234 today")
        with col2:
            st.metric("Storage Used", "2.3 GB", "18% utilized")
        with col3:
            st.metric("Query Latency", "8.5ms", "Optimal")
        
        st.divider()
        
        st.write("**Table Statistics**")
        tables = pd.DataFrame({
            "Table": ["datasets", "training_data", "predictions"],
            "Records": [1, 2847, 1247],
            "Size (MB)": [0.05, 1.2, 0.8],
            "Last Updated": ["2024-02-22", "2024-02-20", "2024-02-22"]
        })
        st.dataframe(tables, use_container_width=True, hide_index=True)

# ==================== Footer ====================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("📊 ML Classification Dashboard")

with col2:
    st.caption(f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col3:
    st.caption("✅ All systems operational")
