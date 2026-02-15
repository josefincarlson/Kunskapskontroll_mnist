import numpy as np
import streamlit as st
import joblib
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


# App-konfig
st.set_page_config(
    page_title="MNIST - använder SVC för att analysera siffror",
    layout="centered"
)


# Ladda modellen
@st.cache_resource
def load_number_model():
    return joblib.load("svc_mnist.joblib")

number_model = load_number_model()
pred = int(number_model.predict(X_input)[0])

st.title("Skriv en siffra mellan 0–9 och få en prediction")

# För att kunna resetta canvas
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

## Fasta canvas-inställningar
canvas_size = 280
stroke_width = 18


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

# Preprocess

def preprocess(canvas_rgba: np.ndarray) -> np.ndarray:
    """
    Tar RGBA-bild från canvas och gör den kompatibel med MNIST/SVC:
    - RGBA -> Gråskala
    - Resize till 28x28
    - Ev invertering om bakgrunden råkar vara ljus
    - Flatten till (1, 784) i skalan 0–255 (float32)
    """
    img = Image.fromarray(canvas_rgba.astype("uint8"), mode="RGBA").convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    arr = np.array(img).astype(np.float32)

    # Om den råkar bli "felvänd": ljus bakgrund och mörk siffra -> invertera
    if arr.mean() > 127:
        img = ImageOps.invert(img)
        arr = np.array(img).astype(np.float32)

    return arr.reshape(1, -1).astype(np.float32)

# Hjälpfunktion, hämtar X_input och förhandsgranskning om det finns
def get_input_and_preview():
    if canvas_result.image_data is None:
        return None, None
    X_input = preprocess(canvas_result.image_data)
    preview = X_input.reshape(28, 28).astype(np.uint8)
    return X_input, preview


# Knappar
col_predict, col_clear = st.columns(2) 

with col_predict:
    if st.button("Analysera", use_container_width=True):
        X_input, _ = get_input_and_preview()
        if X_input is None:
            st.warning("Rita en siffra först.")
        else:
            pred = int(model.predict(X_input)[0])
            st.success(f"Du har skrivit siffra: {pred}")

with col_clear:
    if st.button("Töm ritytan", use_container_width=True):
        st.session_state.canvas_key += 1
        st.rerun()

# Förhandsgranskning
st.subheader("Förhandsgranskning 28×28")  

if preview_img is not None:
    st.image(preview_img, clamp=True, width=140)
else:
    st.caption("Förhandsgranskning visas när du har ritat något.")
