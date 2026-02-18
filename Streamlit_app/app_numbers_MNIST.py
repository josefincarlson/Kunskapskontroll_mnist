import numpy as np
import pandas as pd
import streamlit as st
import joblib
import altair as alt

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


st.title("Skriv en siffra mellan 0–9 och få en prediction")

# För att kunna resetta canvas
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

## Fasta canvas-inställningar
canvas_size = 280
stroke_width = 12


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
def shift_to_center(arr28: np.ndarray) -> np.ndarray:
    arr = arr28.astype(np.float32)
    total = arr.sum()
    if total <= 1e-6:
        return arr28

    ys, xs = np.indices(arr.shape)
    cy = (ys * arr).sum() / total
    cx = (xs * arr).sum() / total

    shift_y = int(round(14 - cy))
    shift_x = int(round(14 - cx))

    shifted = np.zeros_like(arr)
    src_y0 = max(0, -shift_y); src_y1 = min(28, 28 - shift_y)
    dst_y0 = max(0,  shift_y); dst_y1 = min(28, 28 + shift_y)

    src_x0 = max(0, -shift_x); src_x1 = min(28, 28 - shift_x)
    dst_x0 = max(0,  shift_x); dst_x1 = min(28, 28 + shift_x)

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return shifted


def preprocess(canvas_rgba: np.ndarray) -> np.ndarray:
    img = Image.fromarray(canvas_rgba.astype("uint8"), mode="RGBA").convert("L")

    arr0 = np.array(img)
    if arr0.mean() > 127:
        img = ImageOps.invert(img)

    arr = np.array(img)
    thresh = 40
    mask = arr > thresh

    if not mask.any():
        return np.zeros((1, 784), dtype=np.float32)

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    margin = 6  # <--- mindre än 10
    y0 = max(y0 - margin, 0)
    x0 = max(x0 - margin, 0)
    y1 = min(y1 + margin, arr.shape[0] - 1)
    x1 = min(x1 + margin, arr.shape[1] - 1)

    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))

    w, h = cropped.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * (20 / w))))
    else:
        new_h = 20
        new_w = max(1, int(round(w * (20 / h))))

    cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas.paste(cropped, (left, top))

    out = np.array(canvas).astype(np.float32)
    out = shift_to_center(out) 

    return out.reshape(1, -1).astype(np.float32)




# Hjälpfunktion, hämtar X_input och förhandsgranskning om det finns
def get_input_and_preview():
    if canvas_result.image_data is None:
        return None, None
    X_input = preprocess(canvas_result.image_data)
    preview = X_input.reshape(28, 28).astype(np.uint8)
    return X_input, preview

def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)  # stabilitet
    ex = np.exp(x)
    return ex / np.sum(ex)

def get_confidence_scores(model, X_input: np.ndarray) -> np.ndarray:
    """
    Returnerar en vektor med 10 värden (0..9) som summerar till 1.
    - Om modellen har predict_proba => riktiga sannolikheter.
    - Annars => decision_function -> softmax (pseudo-sannolikheter).
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]  # shape (10,)
        return probs.astype(np.float32)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_input)
        # scores kan vara (10,) eller (1,10)
        scores = np.array(scores).reshape(-1)
        return softmax(scores)

    # Fallback (borde inte hända för sklearn-modeller)
    out = np.zeros(10, dtype=np.float32)
    pred = int(model.predict(X_input)[0])
    out[pred] = 1.0
    return out


# Analys
col_predict, col_clear = st.columns(2) 

with col_predict:
    if st.button("Analysera", use_container_width=True):
        X_input, _ = get_input_and_preview()
        if X_input is None:
            st.warning("Rita en siffra först.")
        else:
            scores = get_confidence_scores(number_model, X_input)
            pred = int(np.argmax(scores))
            st.success(f"Du har skrivit siffra: {pred}")

            # Spara senaste scores så vi kan visa staplar längst ner (även efter rerun)
            st.session_state["last_scores"] = scores


with col_clear:
    if st.button("Töm ritytan", use_container_width=True):
        st.session_state.canvas_key += 1
        st.rerun()


# Confidence-staplar (0–9)
st.subheader("Träffsäkerhet %")

scores = st.session_state.get("last_scores", None)

if scores is None:
    st.caption("Klicka på 'Analysera' för att se träffsäkerhet på analysen.")
else:
    df = pd.DataFrame({
        "Siffra": [str(i) for i in range(10)],
        "Confidence": scores
    })

    # ---- AXEL-TEXTER + FORMAT ----
    x_axis = alt.X(
        "Siffra:N",
        sort=[str(i) for i in range(10)],
        title="Träffade siffror",
        axis=alt.Axis(
            labelAngle=0,   # 0 = texten ligger normalt (inte roterad)
            labelFontSize=12
        )
    )

    y_axis = alt.Y(
        "Confidence:Q",
        title="Procentuell träffsäkerhet",
        axis=alt.Axis(format=".0%", tickCount=6),  # 0%, 20%, 40%...
        scale=alt.Scale(domain=[0, 1])
    )

    # ---- STAPLAR ----
    bars = alt.Chart(df).mark_bar().encode(
        x=x_axis,
        y=y_axis,
        tooltip=[
            alt.Tooltip("Siffra:N", title="Siffra"),
            alt.Tooltip("Confidence:Q", title="Träffsäkerhet", format=".1%")
        ]
    ).properties(height=320)

    # ---- PROCENTTEXT OVANFÖR VARJE STAPEL (endast > 0) ----
    df["label_y"] = np.maximum(df["Confidence"] - 0.03, 0.02) 
    df_lbl = df[df["Confidence"] >= 0.01].copy()                 # bara labels för > 0.01

    label_bg = alt.Chart(df_lbl).mark_point(
        shape="square",
        size=800,
        filled=True,
        color="white",
        opacity=1.0
    ).encode(
        x=alt.X("Siffra:N", sort=[str(i) for i in range(10)]),
        y=alt.Y("label_y:Q", scale=alt.Scale(domain=[0, 1]))
    )

    labels = alt.Chart(df_lbl).mark_text(
        fontSize=12,
        color="white",
        stroke="black",
        strokeWidth=1
    ).encode(
        x=alt.X("Siffra:N", sort=[str(i) for i in range(10)]),
        y=alt.Y("label_y:Q", scale=alt.Scale(domain=[0, 1])),
        text=alt.Text("Confidence:Q", format=".1%")
    )

    st.altair_chart(bars + label_bg + labels, use_container_width=True)