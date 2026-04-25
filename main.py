import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# -----------------------------------------
# Page Configuration
# -----------------------------------------
st.set_page_config(
    page_title="Smart Power Forecaster",
    page_icon="⚡",
    layout="wide"
)

# -----------------------------------------
# Custom Styling
# -----------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.metric-value {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Load Model Artifacts (FIXED)
# -----------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        loaded_model = joblib.load('best_model.pkl')
        loaded_scaler = joblib.load('scaler.pkl')
        loaded_selected_features = joblib.load('selected_features.pkl')
        loaded_all_features = joblib.load('all_features.pkl')

        return loaded_model, loaded_scaler, loaded_selected_features, loaded_all_features

    except FileNotFoundError:
        return None, None, None, None

model, scaler, selected_features, all_features = load_artifacts()

# -----------------------------------------
# Gauge Chart
# -----------------------------------------
def create_gauge_chart(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': " kW"},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 2], 'color': "#e6f2fd"},
                {'range': [2, 5], 'color': "#fff3e0"},
                {'range': [5, 10], 'color': "#ffebee"}
            ]
        }
    ))
    fig.update_layout(height=300)
    return fig

# -----------------------------------------
# Sidebar Inputs
# -----------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")

    hour = st.slider("Hour", 0, 23, 12)
    day = st.selectbox("Day", list(range(7)),
                       format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

    lag_1 = st.number_input("Prev Hour Power", value=1.4)
    rolling_mean_24 = st.number_input("24h Avg Power", value=1.3)

# -----------------------------------------
# Main UI
# -----------------------------------------
st.title("⚡ Smart Electricity Forecast")

if model is None:
    st.error("Model files not found. Please train model first.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

with col1:
    reactive = st.number_input("Reactive Power", value=0.1)

with col2:
    voltage = st.number_input("Voltage", value=240.0)

with col3:
    intensity = st.number_input("Intensity", value=5.0)

with col4:
    sub3 = st.number_input("Sub Meter 3", value=18.0)

# -----------------------------------------
# -----------------------------------------
# Prediction Button
# -----------------------------------------
if st.button("🚀 Predict"):

    # Ensure correct types
    hour_val = int(hour)
    day_val = int(day)

    # Input dictionary
    input_data = {
        'Global_reactive_power': reactive,
        'Voltage': voltage,
        'Global_intensity': intensity,
        'Sub_metering_1': 0.0,
        'Sub_metering_2': 0.0,
        'Sub_metering_3': sub3,
        'hour': hour_val,
        'day': day_val,
        'lag_1': lag_1,
        'rolling_mean_24': rolling_mean_24
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # ✅ FIX: Add only missing columns (do NOT overwrite existing ones)
    for col in all_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure correct order
    input_df = input_df[all_features]

    # Scale
    scaled = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled, columns=all_features)

    # Select features
    final_input = scaled_df[selected_features]

    # Predict
    prediction = float(model.predict(final_input)[0])
    prediction = max(0.0, prediction)

    # -----------------------------------------
    # Display
    # -----------------------------------------
    st.subheader("📊 Prediction Result")

    res1, res2 = st.columns([1, 2])

    with res1:
        st.markdown(
            f'<div class="metric-value">{prediction:.2f} kW</div>',
            unsafe_allow_html=True
        )

    with res2:
        st.plotly_chart(create_gauge_chart(prediction), use_container_width=True)

    # Debug
    with st.expander("🔍 Debug Info"):
        st.write("Input:", input_df)
        st.write("Scaled:", scaled_df)
        st.write("Prediction:", prediction)