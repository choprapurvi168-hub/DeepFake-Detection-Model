import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

st.set_page_config(
    page_title="Deepfake Detection",
    layout="centered"
)

st.markdown("""
<style>
body {
    background-color: #0d0f14;
}

.stApp {
    background-color: #0d0f14;
}

.block-container {
    max-width: 900px;
    padding-top: 30px;
    padding-bottom: 50px;
}

.title {
    text-align: center;
    color: white;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 25px;
}

.line {
    height: 1px;
    background: #30333b;
    margin-bottom: 25px;
}

.status {
    background: #123d2c;
    color: #4ade80;
    padding: 14px;
    border-radius: 8px;
    margin: 20px 0;
}

.result-real {
    background: #123d2c;
    color: #4ade80;
    padding: 15px;
    border-radius: 8px;
    font-size: 22px;
    font-weight: bold;
}

.result-fake {
    background: #451d24;
    color: #ff6575;
    padding: 15px;
    border-radius: 8px;
    font-size: 22px;
    font-weight: bold;
}

.bar {
    width: 100%;
    height: 8px;
    background: #292d36;
    border-radius: 10px;
    margin: 8px 0 18px 0;
}

.fill {
    height: 8px;
    background: #2389df;
    border-radius: 10px;
}

.info {
    background: #171a21;
    border: 1px solid #2b2f3a;
    padding: 15px;
    border-radius: 8px;
    margin-top: 15px;
}

h1, h2, h3, p, label {
    color: white !important;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="title"> Deepfake Detection</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="line"></div>', unsafe_allow_html=True)


# LOAD MODEL

MODEL_PATH = Path(__file__).parent / "my_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()

    st.markdown(
        '<div class="status">✅ Model loaded successfully</div>',
        unsafe_allow_html=True
    )

except Exception as e:
    st.error("Model could not be loaded.")
    st.exception(e)
    st.stop()


# FILE TYPE
file_type = st.radio(
    "Select file type to upload",
    ["Video", "Image"],
    horizontal=True
)

st.markdown("---")


# PROBABILITY FUNCTION

def probabilities(prediction):

    prediction = np.asarray(prediction).flatten()

    # Two-class model
    if len(prediction) >= 2:

        fake = float(prediction[0])
        real = float(prediction[1])

        # If output is logits
        if fake < 0 or real < 0 or fake > 1 or real > 1:
            values = np.exp(prediction[:2] - np.max(prediction[:2]))
            values = values / values.sum()
            fake, real = values

        else:
            total = fake + real

            if total > 0:
                fake = fake / total
                real = real / total

        return fake * 100, real * 100

    # Single sigmoid output
    real = float(np.clip(prediction[0], 0, 1))
    fake = 1 - real

    return fake * 100, real * 100


# IMAGE PREDICTION

def predict_image(image):

    img = Image.open(image).convert("RGB")
    img = img.resize((150, 150))

    img = np.array(img)
    img = img.reshape(1, 150, 150, 3)

    prediction = model.predict(img, verbose=0)

    class_id = np.argmax(prediction)

    # Your existing class mapping
    label = "Fake" if class_id == 0 else "Real"

    fake, real = probabilities(prediction)

    return label, fake, real


# VIDEO PREDICTION

def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = 0
    fake_frames = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        total_frames += 1

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (150, 150))

        frame = frame.reshape(1, 150, 150, 3)

        prediction = model.predict(frame, verbose=0)

        class_id = np.argmax(prediction)

        if class_id == 0:
            fake_frames += 1

    cap.release()

    if total_frames == 0:
        return "Real", 0, 100, 0, 0

    fake = fake_frames / total_frames * 100
    real = 100 - fake

    label = "Fake" if fake > 50 else "Real"

    return label, fake, real, total_frames, fake_frames


# RESULT

def show_result(label, fake, real):

    confidence = max(fake, real)

    if label == "Real":
        st.markdown(
            '<div class="result-real">🟢 REAL</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-fake">🔴 FAKE</div>',
            unsafe_allow_html=True
        )

    st.write("### Confidence")
    st.write(f"**{confidence:.2f}%**")

    st.markdown(
        f"""
        <div class="bar">
            <div class="fill" style="width:{confidence:.2f}%"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("### 📊 Prediction Probabilities")

    st.write(f"🔴 **Fake: {fake:.2f}%**")

    st.markdown(
        f"""
        <div class="bar">
            <div class="fill" style="width:{fake:.2f}%"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(f"🟢 **Real: {real:.2f}%**")

    st.markdown(
        f"""
        <div class="bar">
            <div class="fill" style="width:{real:.2f}%"></div>
        </div>
        """,
        unsafe_allow_html=True
    )


# VIDEO INTERFACE

if file_type == "Video":

    st.header("Upload a Video for Deepfake Detection")

    video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi"]
    )

    if video is not None:

        video_path = Path("uploaded_video.mp4")

        with open(video_path, "wb") as file:
            file.write(video.getbuffer())

        left, right = st.columns([1.4, 1])
        with left:
            st.subheader("🎬 Uploaded Video")
            st.video(video)
            st.write(f"**File:** {video.name}")

        with right:

            st.subheader("🔎 Detection")

            if st.button(
                "🔍 Detect Deepfake",
                use_container_width=True
            ):

                with st.spinner("Analyzing video..."):

                    label, fake, real, total, fake_frames = predict_video(
                        str(video_path)
                    )

                show_result(label, fake, real)

                st.markdown(
                    f"""
                    <div class="info">
                    <b>Total Frames:</b> {total}<br>
                    <b>Fake Frames:</b> {fake_frames}<br>
                    <b>Fake Frame Percentage:</b> {fake:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

           

# IMAGE INTERFACE

else:

    st.header("Upload an Image for Deepfake Detection")

    image = st.file_uploader(
        "Choose a JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    if image is not None:

        picture = Image.open(image).convert("RGB")

        # Create two columns
        left, right = st.columns([1.4, 1])

        with left:

            st.subheader("🖼️ Uploaded Image")

            st.image(
                picture,
                caption="Input Image",
                use_container_width=True
            )

            st.write(f"**File:** {image.name}")

        with right:

            st.subheader("🔎 Detection")

            if st.button(
                "🔍 Detect Deepfake",
                use_container_width=True
            ):

                with st.spinner("Analyzing image..."):

                    label, fake, real = predict_image(image)

                show_result(label, fake, real)      