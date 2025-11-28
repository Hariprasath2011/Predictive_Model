import json
import requests
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

PREPROCESSOR_PATH = Path("models/preprocessor.pkl")
MODEL_PATH = Path("models/lgbm.pkl")   

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b"  


@st.cache_resource
def load_preprocessor_and_model():
    """Load the preprocessor and ML model from disk (cached)."""
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    pre = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return pre, model


def call_ollama(prompt: str) -> str:
    """
    Call the local Ollama API with the given prompt and return generated text.
    Uses gemma:2b model and non-streaming response.
    """
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,      
            "temperature": 0.2,
            "max_tokens": 512
        }
        headers = {"Content-Type": "application/json"}
        resp = requests.post(OLLAMA_URL, data=json.dumps(payload), headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        text = data.get("response")
        if text is None:
  
            text = json.dumps(data, indent=2)
        return text

    except Exception as e:
        return f"(AI assistant error: {e})"


def build_explanation_prompt(features: dict, time_days: float, total_cost: float) -> str:
    """
    Build a concise, structured prompt for the AI assistant.
    """
    return f"""
You are an expert assistant in construction project planning and estimation.

The ML model predicted the following for a construction project:

Inputs:
{json.dumps(features, indent=2)}

Model predictions:
- Total time (days): {time_days:.2f}
- Total cost: {total_cost:.2f}

Tasks:
1. Explain in simple, clear language why the time and cost might be at these levels.
2. Highlight the 3 most influential factors from the inputs that likely increased time or cost.
3. Suggest 3 practical recommendations to reduce time and/or cost.
4. Keep the answer short, structured in bullet points.
"""

def main():
    st.set_page_config(
        page_title="Construction Time & Cost Predictor with AI Assistant",
        layout="centered"
    )

    st.title("🏗️ Construction Time & Cost Predictor")

    try:
        preprocessor, model = load_preprocessor_and_model()
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {e}")
        st.stop()

    st.markdown("---")
    st.header("📥 Project Inputs")

    col1, col2 = st.columns(2)

    with col1:
        land_size = st.number_input("Land size (sq.ft)", min_value=100.0, max_value=1_000_000.0, value=5000.0, step=100.0)
        materials_cost = st.number_input("Estimated materials cost", min_value=10000.0, max_value=100_000_000.0, value=100000.0, step=5000.0)
        num_labours = st.number_input("Number of labourers", min_value=1, max_value=1000, value=30, step=1)
        labour_efficiency = st.slider("Labour efficiency (0–1)", min_value=0.1, max_value=1.0, value=0.8, step=0.05)

    with col2:
        terrain = st.selectbox("Terrain type", ["plain", "hilly", "mountainous"])
        project_type = st.selectbox("Project type", ["residential", "commercial", "industrial"])
        weather_index = st.slider("Weather impact index (0 = ideal, 1 = very bad)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        material_shortage_risk = st.slider("Material shortage risk (0–1)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        demand_supply = st.slider("Demand–supply factor (-1 = low demand, 0 = normal, 1 = very high demand)",
                                  min_value=-1.0, max_value=1.0, value=0.0, step=0.1)

    st.markdown("---")

    if st.button(" Predict Time & Cost "):

        features = {
            "land_size": land_size,
            "materials_cost": materials_cost,
            "num_labours": num_labours,
            "labour_efficiency": labour_efficiency,
            "terrain": terrain,
            "project_type": project_type,
            "weather_index": weather_index,
            "material_shortage_risk": material_shortage_risk,
            "demand_supply": demand_supply,
        }

        try:
            input_df = pd.DataFrame([features])
            X = preprocessor.transform(input_df)
            prediction = model.predict(X)[0]  
            time_days = float(prediction[0])
            total_cost = float(prediction[1])
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return

        st.subheader("📊 Prediction")
        col_time, col_cost = st.columns(2)
        with col_time:
            st.metric("Predicted Time (days)", f"{time_days:.1f}")
        with col_cost:
            st.metric("Predicted Total Cost", f"{total_cost:,.2f}")

        st.markdown("---")

        with st.spinner("Asking local AI assistant ..."):
            prompt = build_explanation_prompt(features, time_days, total_cost)
            ai_text = call_ollama(prompt)

        st.subheader("🤖 AI Assistant Explanation")
        st.write(ai_text)

    st.markdown("---")
    st.header("💬 Ask the AI Assistant ")
    user_q = st.text_area(
        "Type any question :",
        placeholder="Example: How does terrain affect my construction time?"
    )
    if st.button("Ask AI Assistant"):
        if not user_q.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Thinking..."):
                q_prompt = (
                    "You are a helpful assistant for a construction time & cost prediction system. "
                    "Answer the question clearly and concisely.\n\n"
                    f"Question: {user_q}"
                )
                answer = call_ollama(q_prompt)
            st.subheader("AI Response")
            st.write(answer)


if __name__ == "__main__":
    main()
