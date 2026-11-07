
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from model import Model

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="ML Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# =====================================
# CSS
# =====================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background-color: #FAFAF8;
    color: #202631;
}

.block-container {
    max-width: 1250px;
    padding-top: 2.5rem;
}

h1, h2, h3 {
    color: #202631;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
}

.subtitle {
    color: #9299A3;
    font-size: 14px;
}

.input-label {
    font-size: 14px;
    font-weight: 700;
    color: #303946;
    margin-bottom: 8px;
}

div[data-testid="stNumberInput"] input {
    background-color: white;
    border-radius: 12px;
    border: 1px solid #E5E9EF;
    color: #202631;
    font-weight: 600;
}

.prediction-card {
    background: linear-gradient(135deg, #E3F1FF, #F1F7FF);
    border: 1px solid #D6E8F9;
    border-radius: 24px;
    padding: 28px;
    min-height: 180px;
}

.metric-card {
    background: white;
    border: 1px solid #ECEAE5;
    border-radius: 22px;
    padding: 22px;
    min-height: 145px;
}

.metric-label {
    color: #9299A3;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #202631;
    font-size: 38px;
    font-weight: 800;
    margin-top: 12px;
}

.prediction-value {
    color: #287CC5;
    font-size: 65px;
    font-weight: 800;
    letter-spacing: -3px;
    margin-top: 8px;
}

div.stButton > button {
    background: #3689CF;
    color: white;
    border-radius: 14px;
    border: none;
    height: 50px;
    font-size: 15px;
    font-weight: 600;
}

div.stButton > button:hover {
    background: #246FAF;
    color: white;
}

#MainMenu, footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():

    model = Model(
        dataset='Student_Performance.csv',
        test_size=0.2,
        random_state=42
    )

    model.split_data()
    model.train()

    metrics = model.evaluate()

    return model, metrics


try:
    model, metrics = load_model()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# =====================================
# HEADER
# =====================================

st.title("ML Student Performance Predictor")

st.markdown(
    '<p class="subtitle">Intelligent academic performance prediction</p>',
    unsafe_allow_html=True
)

st.write("")


# =====================================
# STUDENT INFORMATION
# =====================================

st.subheader("Student Information")

col1, col2, col3 = st.columns(3, gap="large")

with col1:

    st.markdown(
        '<p class="input-label">Study Hours</p>',
        unsafe_allow_html=True
    )

    study_hours = st.number_input(
        "Enter your daily study hours:",
        min_value=0.0,
        max_value=12.0,
        value=5.0,
        step=0.5,
        format="%.2f",
        label_visibility="visible"
    )

with col2:

    st.markdown(
        '<p class="input-label">Previous Scores</p>',
        unsafe_allow_html=True
    )

    previous_scores = st.number_input(
        "Enter your previous scores:",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=0.5,
        format="%.2f",
        label_visibility="visible"
    )

with col3:

    st.markdown(
        '<p class="input-label">Sleep Hours</p>',
        unsafe_allow_html=True
    )

    sleep_hours = st.number_input(
        "Enter your daily sleep hours:",
        min_value=0.0,
        max_value=12.0,
        value=7.0,
        step=0.5,
        format="%.2f",
        label_visibility="visible"
    )


st.write("")


# =====================================
# PREDICTION
# =====================================

if st.button("Calculate Prediction", use_container_width=True):

    result = model.predict(
        study_hours,
        previous_scores,
        sleep_hours
    )

    st.session_state.prediction = float(result)


# =====================================
# PERFORMANCE OVERVIEW
# =====================================

st.subheader("Performance Overview")

prediction = st.session_state.get("prediction", None)

result_col, mae_col, r2_col = st.columns(
    [2, 1, 1],
    gap="medium"
)

with result_col:

    value = f"{prediction:.2f}" if prediction is not None else "--"

    st.markdown(f"""
    <div class="prediction-card">
        <div class="metric-label">
            Predicted Grade
        </div>
        <div class="prediction-value">
            {value}
        </div>
        <div style="color:#7295B5;font-size:13px;">
            Estimated academic performance
        </div>
    </div>
    """, unsafe_allow_html=True)


with mae_col:

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">MAE</div>
        <div class="metric-value">{metrics['MAE']:.2f}</div>
        <div style="color:#9299A3;font-size:12px;">
            Mean Absolute Error
        </div>
    </div>
    """, unsafe_allow_html=True)


with r2_col:

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">R² SCORE</div>
        <div class="metric-value">{metrics['R2']:.2f}</div>
        <div style="color:#9299A3;font-size:12px;">
            Coefficient of Determination
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================
# MODEL EVALUATION
# =====================================

st.write("")
st.subheader("Model Evaluation")

predictions = model.model.predict(model.X_test)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=model.y_test,
    y=predictions,
    mode="markers",
    marker=dict(
        color="#000000",
        size=8,
        opacity=0.7
    ),
    name="Predictions"
))

fig.add_trace(go.Scatter(
    x=[
        model.y_test.min(),
        model.y_test.max()
    ],
    y=[
        model.y_test.min(),
        model.y_test.max()
    ],
    mode="lines",
    line=dict(
        color="#DC2626",
        dash="dash"
    ),
    name="Perfect Prediction"
))

fig.update_layout(
    height=420,
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(
        family="Inter",
        color="#697586"
    ),
    xaxis_title="Actual Performance",
    yaxis_title="Predicted Performance",
    margin=dict(l=20, r=20, t=25, b=20)
)

fig.update_xaxes(
    showgrid=True,
    gridcolor="#F0F0EC",
    tickformat=".2f"
)

fig.update_yaxes(
    showgrid=True,
    gridcolor="#F0F0EC",
    tickformat=".2f"
)

st.plotly_chart(fig, use_container_width=True)


# =====================================
# FOOTER
# =====================================

st.markdown("""
<div style="
    text-align:center;
    color:#A4A8AE;
    font-size:12px;
    padding:25px;
">
    ML STUDENT PERFORMANCE PREDICTOR
    <br>
    LINEAR REGRESSION MODEL
</div>
""", unsafe_allow_html=True)
