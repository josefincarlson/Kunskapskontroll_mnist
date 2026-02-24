# Importerar bibliotek 
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import altair as alt


# Titel på sidan
st.set_page_config(
    page_title="MNIST - SVC sifferanalys",
    layout="centered"
)

# Laddar modellen
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "svc_mnist.joblib"
    return joblib.load(model_path)
    
model = load_model()

if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0

st.markdown(
    """
    <h1 style="text-align:center; margin-bottom: 12px;">
      Rita en siffra mellan 0–9. Modellen analyserar din siffra.
    </h1>
    """,
    unsafe_allow_html=True
)

# Layoutkolumner
left, mid, right = st.columns([1, 2, 1])


# Canvas
canvas_size = 280
stroke_width = 12

with mid:
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=canvas_size,
        height=canvas_size,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state['canvas_key']}",
        
    )

# Preprocessor
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


def deskew_rotate(arr28: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Roterar bilden så huvudaxeln blir mer vertikal (enkel deskew)."""
    img = arr28.astype(np.float32)

    ys, xs = np.where(img > threshold)
    if len(xs) < 20:
        return img

    # centrerade koordinater
    x = xs - xs.mean()
    y = ys - ys.mean()

    cov = np.cov(np.vstack([x, y]))
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)]  # huvudriktning

    angle = np.arctan2(v[1], v[0])
    rot = (np.pi / 2) - angle  # rotera mot vertikal

    max_rot = np.deg2rad(25)
    rot = float(np.clip(rot, -max_rot, max_rot))

    out = np.zeros_like(img)
    cy, cx = 13.5, 13.5
    cosr, sinr = np.cos(rot), np.sin(rot)

    # nearest-neighbor rotation (28x28)
    for yy in range(28):
        for xx in range(28):
            y0 = yy - cy
            x0 = xx - cx
            xs0 =  cosr * x0 + sinr * y0 + cx
            ys0 = -sinr * x0 + cosr * y0 + cy
            xi, yi = int(round(xs0)), int(round(ys0))
            if 0 <= yi < 28 and 0 <= xi < 28:
                out[yy, xx] = img[yi, xi]

    return out



def preprocess(canvas_rgba: np.ndarray) -> np.ndarray:
    img = Image.fromarray(canvas_rgba.astype("uint8"), mode="RGBA").convert("L")

    arr0 = np.array(img)
    if arr0.mean() > 127:
        img = ImageOps.invert(img)

    arr = np.array(img)
    thresh = 25
    mask = arr > thresh

    if not mask.any():
        return np.zeros((1, 784), dtype=np.float32)

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    margin = 6
    y0 = max(y0 - margin, 0)
    x0 = max(x0 - margin, 0)
    y1 = min(y1 + margin, arr.shape[0] - 1)
    x1 = min(x1 + margin, arr.shape[1] - 1)

    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))

    TARGET = 20

    w, h = cropped.size
    if w > h:
        new_w = TARGET
        new_h = max(1, int(round(h * (TARGET / w))))
    else:
        new_h = TARGET
        new_w = max(1, int(round(w * (TARGET / h))))

    cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas_img = Image.new("L", (28, 28), color=0)
    left_pad = (28 - new_w) // 2
    top_pad = (28 - new_h) // 2
    canvas_img.paste(cropped, (left_pad, top_pad))

    out = np.array(canvas_img).astype(np.float32)

    out = shift_to_center(out)

    return out.reshape(1, -1).astype(np.float32)

def get_input_and_preview():
    if canvas.image_data is None:
        return None, None
    X_input = preprocess(canvas.image_data)
    if np.max(X_input) == 0:
        return None, None
    preview = X_input.reshape(28, 28).astype(np.uint8)
    return X_input, preview

def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def get_confidence_scores(model, X_input: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
        return probs.astype(np.float32)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_input)
        scores = np.array(scores).reshape(-1)
        return softmax(scores)

    out = np.zeros(10, dtype=np.float32)
    pred_ = int(model.predict(X_input)[0])
    out[pred_] = 1.0
    return out

# Auto-analys
X_input, preview = get_input_and_preview()

with mid:
    if X_input is None:
        st.session_state["last_scores"] = None
    else:
        scores = get_confidence_scores(model, X_input)
        pred = int(np.argmax(scores))
        st.session_state["last_scores"] = scores

        # Prediktionskort
        st.markdown(
            f"""
            <div style="display:flex; justify-content:center; margin: 12px 0 6px 0;">
              <div style="width:{canvas_size}px; display:flex; justify-content:center;">
                <div style="
                    padding: 18px 18px;
                    border-radius: 18px;
                    border: 1px solid rgba(255,255,255,0.15);
                    background: rgba(255,255,255,0.06);
                    text-align:center;
                    width: 165px;">
                  <div style="font-size: 18px; opacity: 0.8; letter-spacing: 0.3px;">
                    Prediktion
                  </div>
                  <div style="font-size: 92px; font-weight: 800; line-height: 1;">
                    {pred}
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Rensa-knapp
    btn_l, btn_m, btn_r = st.columns([1, 2, 1])
    with btn_m:
        if st.button("Rensa ritytan", use_container_width=True):
            st.session_state["canvas_key"] += 1
            st.session_state["last_scores"] = None
            st.rerun()

# Staplar för vilka siffror modellen träffar
st.subheader("Modellens säkerhet per siffra (%)")

scores = st.session_state.get("last_scores", None)

if scores is None:
    st.caption("Rita en siffra för att se träffsäkerhet.")
else:
    df = pd.DataFrame({
        "Siffra": [str(i) for i in range(10)],
        "Confidence": scores
    })

    y_max = float(min(1.0, df["Confidence"].max() + 0.10))

    x_axis = alt.X(
    "Siffra:N",
    sort=[str(i) for i in range(10)],
    title="Träffade siffror",
    axis=alt.Axis(labelAngle=0, labelFontSize=12)
    )

    y_axis = alt.Y(
        "Confidence:Q",
        title="Sannolikhet",
        axis=alt.Axis(format=".0%", tickCount=6),
        scale=alt.Scale(domain=[0, y_max])
    )

    bars = alt.Chart(df).mark_bar().encode(
        x=x_axis,
        y=y_axis,
        tooltip=[
            alt.Tooltip("Siffra:N", title="Siffra"),
            alt.Tooltip("Confidence:Q", title="Träffsäkerhet", format=".1%")
        ]
    ).properties(height=320)

    df_lbl = df[df["Confidence"] >= 0.05].copy()

    labels = alt.Chart(df_lbl).mark_text(
        fontSize=13,
        fontWeight="bold",
        dy=-16,
        color="white"
    ).encode(
        x=alt.X("Siffra:N", sort=[str(i) for i in range(10)]),
        y=y_axis,
        text=alt.Text("Confidence:Q", format=".1%")
    )

    chart = (bars + labels).configure_view(
        strokeWidth=0
    ).configure_axis(
        labelColor="white",
        titleColor="white",
        gridColor="rgba(255,255,255,0.15)",
        domainColor="rgba(255,255,255,0.35)",
        tickColor="rgba(255,255,255,0.35)"
    )

    st.altair_chart(chart, use_container_width=True)
