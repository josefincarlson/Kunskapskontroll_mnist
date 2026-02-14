import numpy as np
import streamlit as st
import joblib
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST - using SVC to predict", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("svc_mnist.joblib")

model = load_model()

st.title("Rita en siffra (0–9) och få en prediction")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0


canvas_size = 280
stroke_width = st.slider("Pennbredd", 5, 30, 18)

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=canvas_size,
    height=canvas_size,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",
)

def preprocess(canvas_rgba: np.ndarray) -> np.ndarray:
    img = Image.fromarray(canvas_rgba.astype("uint8"), mode="RGBA").convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    arr = np.array(img).astype(np.float32)

    # Om bakgrunden råkar bli ljus, invertera
    if arr.mean() > 127:
        img = ImageOps.invert(img)
        arr = np.array(img).astype(np.float32)

    # Till (1, 784) i 0–255 (matchar din träning)
    return arr.reshape(1, -1).astype(np.float32)

col1, col2 = st.columns(2)

with col1:
    if st.button("Predict"):
        if canvas_result.image_data is None:
            st.warning("Rita en siffra först.")
        else:
            X_input = preprocess(canvas_result.image_data)
            pred = int(model.predict(X_input)[0])
            st.success(f"Prediction: {pred}")

with col2:
    if st.button("Clear"):
        st.session_state.canvas_key += 1
        st.rerun()s

st.subheader("Preview 28×28")
if canvas_result.image_data is not None:
    X_input = preprocess(canvas_result.image_data)
    preview = X_input.reshape(28, 28).astype(np.uint8)
    st.image(preview, clamp=True, width=140)
