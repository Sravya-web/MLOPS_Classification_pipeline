"""Professional Streamlit dashboard for hepatitis classification."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).resolve().parent
LOGO_PATH = FRONTEND_DIR / "image_11.jpeg"
DATA_PATH = ROOT_DIR / "data" / "hepatitis.csv"
MODEL_PATH = ROOT_DIR / "model_artifacts" / "hepatitis_model.joblib"
PREPROCESSOR_PATH = ROOT_DIR / "model_artifacts" / "hepatitis_preprocessor.joblib"

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = 5

FEATURES = [
    "Age",
    "Sex",
    "ALB",
    "ALP",
    "ALT",
    "AST",
    "BIL",
    "CHE",
    "CHOL",
    "CREA",
    "GGT",
    "PROT",
]

LAB_FEATURES = ["ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]

CATEGORY_LABELS = {
    0: "Blood Donor",
    1: "Hepatitis",
    2: "Fibrosis",
    3: "Cirrhosis",
    4: "Suspect Blood Donor",
}

CATEGORY_COLORS = {
    "0=Blood Donor": "#16a34a",
    "0s=suspect Blood Donor": "#64748b",
    "1=Hepatitis": "#f97316",
    "2=Fibrosis": "#eab308",
    "3=Cirrhosis": "#dc2626",
    "Blood Donor": "#16a34a",
    "Suspect Blood Donor": "#64748b",
    "Hepatitis": "#f97316",
    "Fibrosis": "#eab308",
    "Cirrhosis": "#dc2626",
}


st.set_page_config(
    page_title="HepatoAI Clinical Intelligence",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    """Apply the custom healthcare interface styling."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f9fb;
            --surface: #ffffff;
            --surface-soft: #eef8f7;
            --ink: #102033;
            --muted: #64748b;
            --line: #dbe7ee;
            --primary: #0f766e;
            --primary-dark: #115e59;
            --accent: #2563eb;
            --danger: #dc2626;
            --warning: #d97706;
            --success: #16a34a;
        }

        html, body, [class*="css"] {
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(20, 184, 166, .13), transparent 34rem),
                linear-gradient(180deg, #f8fcfd 0%, var(--bg) 38%, #f7fafc 100%);
        }

        .main .block-container {
            max-width: 1380px;
            padding: 2rem 2.25rem 3rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f2f3a 0%, #123c44 48%, #0b2530 100%);
            border-right: 1px solid rgba(255,255,255,.08);
        }

        section[data-testid="stSidebar"] * {
            color: rgba(255,255,255,.92);
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
            background: rgba(255,255,255,.07);
            border: 1px solid rgba(255,255,255,.1);
            border-radius: 10px;
            padding: .55rem .7rem;
            margin: .28rem 0;
        }

        section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
            background: rgba(255,255,255,.13);
        }

        section[data-testid="stSidebar"] .stTextInput input {
            background: rgba(255,255,255,.08);
            border-color: rgba(255,255,255,.18);
            color: #ffffff;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        h1 {
            font-size: clamp(2rem, 4vw, 3.35rem);
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: .65rem;
        }

        h2 {
            font-size: 1.3rem;
            font-weight: 750;
            margin-top: .25rem;
        }

        .eyebrow {
            color: var(--primary);
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin-bottom: .7rem;
        }

        .hero {
            background: linear-gradient(135deg, rgba(255,255,255,.95), rgba(235,249,247,.94));
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 2rem;
            box-shadow: 0 18px 50px rgba(15, 45, 58, .08);
            margin-bottom: 1.25rem;
        }

        .hero p {
            max-width: 820px;
            color: var(--muted);
            font-size: 1.04rem;
            line-height: 1.65;
            margin: 0;
        }

        .metric-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 32px rgba(15, 45, 58, .06);
            min-height: 126px;
        }

        .metric-label {
            color: var(--muted);
            font-size: .78rem;
            font-weight: 750;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .metric-value {
            color: var(--ink);
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1.2;
            margin-top: .35rem;
        }

        .metric-note {
            color: var(--muted);
            font-size: .86rem;
            margin-top: .35rem;
        }

        .panel {
            background: rgba(255,255,255,.92);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.2rem;
            box-shadow: 0 12px 30px rgba(15, 45, 58, .055);
            margin-bottom: 1rem;
        }

        .result-card {
            border-radius: 8px;
            padding: 1.35rem;
            border: 1px solid var(--line);
            background: linear-gradient(135deg, #ffffff, #eef8f7);
            box-shadow: 0 12px 32px rgba(15, 45, 58, .06);
        }

        .risk-low { border-left: 6px solid var(--success); }
        .risk-watch { border-left: 6px solid var(--warning); }
        .risk-high { border-left: 6px solid var(--danger); }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            border-radius: 999px;
            padding: .34rem .68rem;
            font-size: .78rem;
            font-weight: 750;
            border: 1px solid transparent;
        }

        .status-ok {
            color: #065f46;
            background: #dcfce7;
            border-color: #bbf7d0;
        }

        .status-bad {
            color: #991b1b;
            background: #fee2e2;
            border-color: #fecaca;
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 10px 28px rgba(15, 45, 58, .055);
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid var(--primary);
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            font-weight: 800;
            min-height: 2.8rem;
            box-shadow: 0 10px 24px rgba(15, 118, 110, .22);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: .35rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: .6rem 1rem;
            background: #eaf5f4;
        }

        .block-spacer {
            height: .45rem;
        }

        @media (max-width: 900px) {
            .main .block-container {
                padding: 1.2rem 1rem 2rem;
            }

            .hero {
                padding: 1.25rem;
            }

            .metric-card {
                min-height: auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def configure_plotly(fig: go.Figure, height: int = 390) -> go.Figure:
    """Give Plotly charts a consistent clinical analytics style."""
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.7)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#102033"),
        margin=dict(l=28, r=24, t=54, b=34),
        title=dict(font=dict(size=18, color="#102033")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e7eef3", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e7eef3", zeroline=False)
    return fig


@st.cache_data(ttl=300)
def load_hepatitis_data() -> pd.DataFrame | None:
    """Load the hepatitis dataset from a repo-relative path."""
    try:
        df = pd.read_csv(DATA_PATH)
        df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")], errors="ignore")
        df = df.dropna(subset=["Category"])
        return df
    except Exception:
        return None


@st.cache_resource
def load_model_assets():
    """Load the trained hepatitis model and preprocessor."""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception:
        return None, None


def check_api_health(api_url: str) -> dict[str, object]:
    """Check the optional FastAPI service status."""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=API_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        return {"healthy": True, "model_loaded": payload.get("model_loaded", False)}
    except Exception:
        return {"healthy": False, "model_loaded": False}


def metric_card(label: str, value: str | int | float, note: str = "") -> None:
    """Render a compact, responsive metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, body: str, eyebrow: str = "Clinical AI Platform") -> None:
    """Render a high-impact page header."""
    st.markdown(
        f"""
        <section class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{body}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def category_name(raw_category: str) -> str:
    """Convert dataset category strings to clean labels."""
    if "=" in raw_category:
        return raw_category.split("=", 1)[1].replace("suspect", "Suspect").strip()
    return raw_category


def risk_class(category: str) -> str:
    """Return the visual risk class for a predicted category."""
    if category in {"Cirrhosis", "Hepatitis"}:
        return "risk-high"
    if category in {"Fibrosis", "Suspect Blood Donor"}:
        return "risk-watch"
    return "risk-low"


def prepare_features(values: dict[str, float | int | str]) -> np.ndarray:
    """Prepare model features in the order used during training."""
    encoded = []
    for feature in FEATURES:
        if feature == "Sex":
            encoded.append(0 if values.get("Sex") == "m" else 1)
        else:
            encoded.append(float(values.get(feature, 0)))
    return np.array(encoded).reshape(1, -1)


def predict_patient(values: dict[str, float | int | str]) -> tuple[str, float, np.ndarray] | None:
    """Run the local model prediction and return label, confidence, probabilities."""
    model, preprocessor = load_model_assets()
    if model is None or preprocessor is None:
        return None

    features = prepare_features(values)
    scaled_features = preprocessor.transform(features)
    prediction = int(model.predict(scaled_features)[0])
    probabilities = model.predict_proba(scaled_features)[0]
    return CATEGORY_LABELS.get(prediction, "Unknown"), float(probabilities[prediction]), probabilities


def build_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["Category"].value_counts()
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[category_name(category) for category in counts.index],
                values=counts.values,
                hole=0.62,
                marker=dict(colors=[CATEGORY_COLORS.get(category, "#0f766e") for category in counts.index]),
                textinfo="percent",
            )
        ]
    )
    fig.update_traces(hovertemplate="<b>%{label}</b><br>%{value} patients<br>%{percent}<extra></extra>")
    fig.update_layout(title="")
    return configure_plotly(fig, height=410)


def build_age_chart(df: pd.DataFrame) -> go.Figure:
    clean = df.copy()
    clean["Clinical Category"] = clean["Category"].map(category_name)
    fig = px.box(
        clean,
        x="Clinical Category",
        y="Age",
        color="Clinical Category",
        color_discrete_map=CATEGORY_COLORS,
        points="outliers",
        title="Age Distribution by Clinical Category",
    )
    fig.update_layout(showlegend=False)
    return configure_plotly(fig, height=410)


def sidebar(df: pd.DataFrame | None) -> tuple[str, str]:
    """Render sidebar navigation and return selected page and API URL."""
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width="stretch")
        else:
            st.markdown("### HepatoAI")

        st.markdown("## HepatoAI")
        st.caption("Classification intelligence for liver panel review.")

        page = st.radio(
            "Workspace",
            [
                "Command Center",
                "Patient Triage",
                "Batch Review",
                "Population Analytics",
                "Feature Explorer",
                "Clinical Guide",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### Integrations")
        api_url = st.text_input("Prediction API", value=API_URL, help="Optional FastAPI service health check.")
        health = check_api_health(api_url)
        status_class = "status-ok" if health["healthy"] else "status-bad"
        status_text = "API online" if health["healthy"] else "API offline"
        st.markdown(f'<span class="status-pill {status_class}">{status_text}</span>', unsafe_allow_html=True)

        st.divider()
        st.markdown("### Data Snapshot")
        if df is None:
            st.warning("Dataset unavailable")
        else:
            st.metric("Patients", f"{len(df):,}")
            st.metric("Clinical classes", df["Category"].nunique())
            model, preprocessor = load_model_assets()
            model_ready = model is not None and preprocessor is not None
            model_class = "status-ok" if model_ready else "status-bad"
            model_text = "Model ready" if model_ready else "Model missing"
            st.markdown(f'<span class="status-pill {model_class}">{model_text}</span>', unsafe_allow_html=True)

    return page, api_url


def render_overview(df: pd.DataFrame | None) -> None:
    hero(
        "Hepatitis Classification Intelligence",
        "A clinical decision support dashboard for reviewing liver-panel patterns, patient-level risk signals, and population health trends across hepatitis-related categories.",
    )

    if df is None:
        st.error(f"Dataset not found at {DATA_PATH}")
        return

    healthy = int((df["Category"] == "0=Blood Donor").sum())
    elevated = int((df["Category"].isin(["1=Hepatitis", "2=Fibrosis", "3=Cirrhosis"])).sum())
    suspect = int((df["Category"] == "0s=suspect Blood Donor").sum())
    avg_age = f"{df['Age'].mean():.1f}"

    cols = st.columns(4)
    with cols[0]:
        metric_card("Patient records", f"{len(df):,}", "Curated hepatitis dataset")
    with cols[1]:
        metric_card("Blood donors", f"{healthy:,}", f"{healthy / len(df):.1%} of records")
    with cols[2]:
        metric_card("Disease signals", f"{elevated:,}", "Hepatitis, fibrosis, cirrhosis")
    with cols[3]:
        metric_card("Average age", avg_age, f"{suspect} suspect donor records")

    st.markdown('<div class="block-spacer"></div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(build_distribution_chart(df), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(build_age_chart(df), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("## Clinical Dataset")
    sample = df.head(12).copy()
    sample["Clinical Category"] = sample["Category"].map(category_name)
    st.dataframe(sample[["Clinical Category", "Age", "Sex", *LAB_FEATURES]], width="stretch", hide_index=True)


def render_patient_triage(df: pd.DataFrame | None) -> None:
    hero(
        "Patient Triage",
        "Enter demographics and laboratory markers to generate a model-backed hepatitis classification with calibrated category probabilities.",
        "Point-of-Care Analysis",
    )

    model, preprocessor = load_model_assets()
    if model is None or preprocessor is None:
        st.error(f"Model artifacts were not found at {MODEL_PATH.parent}. Train the hepatitis model before running triage.")
        return

    with st.form("patient_triage_form"):
        st.markdown("## Patient Profile")
        profile_cols = st.columns([1, 1, 2])
        with profile_cols[0]:
            age = st.number_input("Age", min_value=18, max_value=100, value=45, step=1)
        with profile_cols[1]:
            sex = st.selectbox("Sex", ["m", "f"], format_func=lambda value: "Male" if value == "m" else "Female")
        with profile_cols[2]:
            st.info("This model is intended for classification support and does not replace clinician review.")

        st.markdown("## Laboratory Markers")
        c1, c2, c3 = st.columns(3)
        with c1:
            alb = st.number_input("ALB - Albumin", min_value=10.0, max_value=85.0, value=40.0, step=0.1)
            alp = st.number_input("ALP - Alkaline Phosphatase", min_value=10.0, max_value=450.0, value=70.0, step=0.1)
            alt = st.number_input("ALT - Alanine Aminotransferase", min_value=0.0, max_value=250.0, value=30.0, step=0.1)
        with c2:
            ast = st.number_input("AST - Aspartate Aminotransferase", min_value=0.0, max_value=350.0, value=35.0, step=0.1)
            bil = st.number_input("BIL - Bilirubin", min_value=0.0, max_value=250.0, value=10.0, step=0.1)
            che = st.number_input("CHE - Cholinesterase", min_value=1.0, max_value=20.0, value=7.0, step=0.1)
        with c3:
            chol = st.number_input("CHOL - Cholesterol", min_value=1.0, max_value=12.0, value=5.0, step=0.1)
            crea = st.number_input("CREA - Creatinine", min_value=5.0, max_value=250.0, value=80.0, step=0.1)
            ggt = st.number_input("GGT - Gamma-Glutamyl Transferase", min_value=0.0, max_value=800.0, value=30.0, step=0.1)
            prot = st.number_input("PROT - Protein", min_value=35.0, max_value=100.0, value=72.0, step=0.1)

        submitted = st.form_submit_button("Analyze Patient", width="stretch", type="primary")

    if not submitted:
        return

    values = {
        "Age": age,
        "Sex": sex,
        "ALB": alb,
        "ALP": alp,
        "ALT": alt,
        "AST": ast,
        "BIL": bil,
        "CHE": che,
        "CHOL": chol,
        "CREA": crea,
        "GGT": ggt,
        "PROT": prot,
    }
    result = predict_patient(values)
    if result is None:
        st.error("Prediction could not be generated.")
        return

    predicted_category, confidence, probabilities = result
    result_col, chart_col = st.columns([0.9, 1.4])
    with result_col:
        st.markdown(
            f"""
            <div class="result-card {risk_class(predicted_category)}">
                <div class="metric-label">Predicted Category</div>
                <div class="metric-value">{predicted_category}</div>
                <div class="metric-note">Confidence: {confidence:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Review Notes")
        if predicted_category == "Blood Donor":
            st.success("Model output is consistent with a lower-risk donor-like profile.")
        elif predicted_category in {"Hepatitis", "Cirrhosis"}:
            st.warning("Elevated-risk classification. Correlate with clinical history and confirmatory testing.")
        else:
            st.info("Intermediate or suspect profile. Consider follow-up evaluation and repeat labs.")

    with chart_col:
        probability_df = pd.DataFrame(
            {
                "Category": [CATEGORY_LABELS.get(i, f"Class {i}") for i in range(len(probabilities))],
                "Probability": probabilities,
            }
        )
        fig = px.bar(
            probability_df,
            x="Probability",
            y="Category",
            orientation="h",
            color="Category",
            color_discrete_map=CATEGORY_COLORS,
            title="Classification Probability Profile",
        )
        fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:.1%}<extra></extra>")
        st.plotly_chart(configure_plotly(fig, height=350), width="stretch")

    if df is not None:
        st.markdown("## Patient Values Compared With Cohort Median")
        median_df = df[LAB_FEATURES].median(numeric_only=True)
        comparison_df = pd.DataFrame(
            {
                "Marker": LAB_FEATURES,
                "Patient": [values[feature] for feature in LAB_FEATURES],
                "Cohort median": [median_df[feature] for feature in LAB_FEATURES],
            }
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(x=comparison_df["Marker"], y=comparison_df["Patient"], name="Patient", marker_color="#0f766e"))
        fig.add_trace(
            go.Scatter(
                x=comparison_df["Marker"],
                y=comparison_df["Cohort median"],
                name="Cohort median",
                mode="lines+markers",
                line=dict(color="#2563eb", width=3),
            )
        )
        fig.update_layout(title="Lab Marker Comparison", yaxis_title="Value")
        st.plotly_chart(configure_plotly(fig, height=390), width="stretch")


def render_batch_review() -> None:
    hero(
        "Batch Review",
        "Upload a CSV with hepatitis lab markers to classify multiple patients and export a clean review file for downstream validation.",
        "Operational Workflow",
    )

    model, preprocessor = load_model_assets()
    if model is None or preprocessor is None:
        st.error("Batch predictions require the local hepatitis model and preprocessor artifacts.")
        return

    uploaded_file = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Required columns: Age, Sex, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT.")
        return

    uploaded_df = pd.read_csv(uploaded_file)
    missing = [column for column in FEATURES if column not in uploaded_df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.dataframe(uploaded_df.head(20), width="stretch", hide_index=True)
        return

    if st.button("Run Batch Classification", width="stretch", type="primary"):
        results = []
        for _, row in uploaded_df.iterrows():
            row_values = {feature: row[feature] for feature in FEATURES}
            prediction = predict_patient(row_values)
            if prediction is None:
                continue
            label, confidence, probabilities = prediction
            results.append(
                {
                    "Prediction": label,
                    "Confidence": confidence,
                    **{f"Probability {CATEGORY_LABELS.get(i, i)}": prob for i, prob in enumerate(probabilities)},
                }
            )

        results_df = pd.concat([uploaded_df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
        st.success(f"Classified {len(results_df):,} patient records.")
        st.dataframe(results_df, width="stretch", hide_index=True)
        st.download_button(
            "Download Classified CSV",
            data=results_df.to_csv(index=False),
            file_name=f"hepatitis_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width="stretch",
        )


def render_population_analytics(df: pd.DataFrame | None) -> None:
    hero(
        "Population Analytics",
        "Explore cohort-level disease patterns, missingness, and marker relationships across clinical categories.",
        "Clinical Operations",
    )

    if df is None:
        st.error("Dataset unavailable.")
        return

    tab1, tab2, tab3 = st.tabs(["Cohort Trends", "Correlations", "Data Quality"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            sex_df = df.groupby(["Category", "Sex"]).size().reset_index(name="Patients")
            sex_df["Clinical Category"] = sex_df["Category"].map(category_name)
            fig = px.bar(
                sex_df,
                x="Clinical Category",
                y="Patients",
                color="Sex",
                barmode="group",
                title="Sex Distribution by Category",
                color_discrete_sequence=["#0f766e", "#60a5fa"],
            )
            st.plotly_chart(configure_plotly(fig), width="stretch")
        with c2:
            clean = df.copy()
            clean["Clinical Category"] = clean["Category"].map(category_name)
            fig = px.violin(
                clean,
                x="Clinical Category",
                y="ALT",
                color="Clinical Category",
                box=True,
                color_discrete_map=CATEGORY_COLORS,
                title="ALT Distribution",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(configure_plotly(fig), width="stretch")

    with tab2:
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        corr = numeric_df.corr()
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="BrBG",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="r"),
            )
        )
        fig.update_layout(title="Laboratory Marker Correlation Matrix")
        st.plotly_chart(configure_plotly(fig, height=560), width="stretch")

    with tab3:
        missing = df.isna().sum().sort_values(ascending=False)
        missing_df = missing[missing > 0].reset_index()
        missing_df.columns = ["Field", "Missing Values"]
        if missing_df.empty:
            st.success("No missing values detected in the loaded dataset.")
        else:
            fig = px.bar(
                missing_df,
                x="Missing Values",
                y="Field",
                orientation="h",
                title="Missing Values by Field",
                color="Missing Values",
                color_continuous_scale="Teal",
            )
            st.plotly_chart(configure_plotly(fig), width="stretch")
        st.dataframe(df.describe(include="all").transpose(), width="stretch")


def render_feature_explorer(df: pd.DataFrame | None) -> None:
    hero(
        "Feature Explorer",
        "Inspect how individual markers behave across clinical categories and compare the descriptive statistics behind each distribution.",
        "Marker Intelligence",
    )

    if df is None:
        st.error("Dataset unavailable.")
        return

    selected_feature = st.selectbox("Marker", ["Age", *LAB_FEATURES])
    clean = df.copy()
    clean["Clinical Category"] = clean["Category"].map(category_name)

    left, right = st.columns(2)
    with left:
        fig = px.box(
            clean,
            x="Clinical Category",
            y=selected_feature,
            color="Clinical Category",
            color_discrete_map=CATEGORY_COLORS,
            title=f"{selected_feature} by Category",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(configure_plotly(fig), width="stretch")
    with right:
        fig = px.histogram(
            clean,
            x=selected_feature,
            color="Clinical Category",
            marginal="rug",
            opacity=0.75,
            color_discrete_map=CATEGORY_COLORS,
            title=f"{selected_feature} Distribution",
        )
        st.plotly_chart(configure_plotly(fig), width="stretch")

    stats = (
        clean.groupby("Clinical Category")[selected_feature]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(2)
        .reset_index()
    )
    st.markdown("## Summary Statistics")
    st.dataframe(stats, width="stretch", hide_index=True)


def render_clinical_guide(df: pd.DataFrame | None) -> None:
    hero(
        "Clinical Guide",
        "A compact reference for interpreting model categories, common marker patterns, and recommended review posture.",
        "Clinical Context",
    )

    guide = pd.DataFrame(
        [
            {
                "Category": "Blood Donor",
                "Signal": "Donor-like lab profile",
                "Review Posture": "Routine review and standard clinical correlation",
            },
            {
                "Category": "Suspect Blood Donor",
                "Signal": "Borderline donor profile",
                "Review Posture": "Repeat labs or confirmatory assessment if clinically indicated",
            },
            {
                "Category": "Hepatitis",
                "Signal": "Potential active hepatic inflammation",
                "Review Posture": "Correlate with viral markers, symptoms, and liver enzyme trend",
            },
            {
                "Category": "Fibrosis",
                "Signal": "Pattern compatible with liver tissue scarring",
                "Review Posture": "Monitor progression and evaluate underlying etiology",
            },
            {
                "Category": "Cirrhosis",
                "Signal": "Advanced liver disease pattern",
                "Review Posture": "Specialist review, complication screening, and longitudinal monitoring",
            },
        ]
    )
    st.dataframe(guide, width="stretch", hide_index=True)

    if df is None:
        return

    st.markdown("## Marker Snapshot")
    selected = st.selectbox("Clinical category", sorted(df["Category"].unique()), format_func=category_name)
    category_df = df[df["Category"] == selected]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Avg ALT", f"{category_df['ALT'].mean():.1f}", "Alanine aminotransferase")
    with c2:
        metric_card("Avg AST", f"{category_df['AST'].mean():.1f}", "Aspartate aminotransferase")
    with c3:
        metric_card("Avg BIL", f"{category_df['BIL'].mean():.1f}", "Bilirubin")
    with c4:
        metric_card("Avg GGT", f"{category_df['GGT'].mean():.1f}", "Gamma-glutamyl transferase")


def render_footer() -> None:
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("HepatoAI Clinical Decision Support")
    with c2:
        st.caption(f"Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with c3:
        st.caption("For research and clinician-assisted review")


apply_theme()
data = load_hepatitis_data()
selected_page, _selected_api_url = sidebar(data)

if selected_page == "Command Center":
    render_overview(data)
elif selected_page == "Patient Triage":
    render_patient_triage(data)
elif selected_page == "Batch Review":
    render_batch_review()
elif selected_page == "Population Analytics":
    render_population_analytics(data)
elif selected_page == "Feature Explorer":
    render_feature_explorer(data)
elif selected_page == "Clinical Guide":
    render_clinical_guide(data)

render_footer()
