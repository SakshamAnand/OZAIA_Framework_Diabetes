# streamlit_app.py

import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image

# -------------------------
# App Setup
# -------------------------
st.set_page_config(page_title="OZAIA Framework", layout="centered")
st.title("OZAIA Net Framework")

# Sidebar: Image and Placeholder Text
with st.sidebar:
    st.image("OZAIA_LOGO.png", use_container_width=True)
    st.markdown(f"OZAIANet is an innovative deep learning framework designed for automated diabetes detection using plantar thermographic images. The system employs an ensemble of convolutional neural networks DenseNet121 and EfficientNetV2B0 to analyze thermal patterns in foot sole images, achieving 94.34% accuracy in distinguishing between diabetic and non-diabetic individuals. \n\nInput Requirements: Infrared thermographic images of foot soles (plantar surface) in standard thermal imaging formats \n\nThis project was developed by Saksham Anand in collaboration with FDDI - VIT Chennai, under the guidance of Dr. Suganya R and Dr. Vimudha M.") 

st.markdown("""
#### Diabetic Foot Thermograph Analyzer
This tool predicts whether a given foot thermograph shows signs of diabetic neuropathy (DM) or is healthy (CTRL). It also generates LIME-based heatmaps to explain the model's decision.

⚠️ **Disclaimer**: This tool is for research and educational purposes only. It is not intended for clinical diagnosis. Please consult a medical professional for actual healthcare decisions.
""")
st.image("instruct.png", use_container_width=True)
# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_trained_model():
    return load_model("OZAIA_Net_DNENV2_final_3.h5")

model = load_trained_model()

# -------------------------
# Prediction Function for LIME
# -------------------------
def predict_fn(images):
    probs = model.predict(np.array(images))
    return np.hstack([1 - probs, probs])  # for binary classification

# -------------------------
# Image Upload
# -------------------------
uploaded_file = st.file_uploader("Upload a foot thermograph image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert('RGB')
    img_resized = img_pil.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0

    st.image(img_resized, caption="Uploaded Image", use_container_width=True)

    pred_prob = model.predict(np.expand_dims(img_array, axis=0))[0][0]
    pred_class = 'DM' if pred_prob >= 0.5 else 'CTRL'
    confidence = pred_prob if pred_class == 'DM' else 1 - pred_prob

    st.markdown(f"### Prediction: `{pred_class}` with confidence **{confidence:.2f}**")

    # Low confidence warning
    if confidence < 0.3:
        st.warning("""⚠️ **Low Confidence Alert**  
Make sure you have uploaded a thermal image of your **foot plantar region**.  
Low confidence score indicates that the model is either receiving poor quality/incomplete data or is unable to make a reliable prediction.  
Clinical validation is **absolutely necessary** in such cases.
""")

    # LIME Explanation
    with st.spinner("Generating LIME Heatmap..."):
        time.sleep(0.5)
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image=img_array.astype('double'),
            classifier_fn=predict_fn,
            hide_color=0,
            num_samples=600
        )

        label = explanation.top_labels[0]
        segments = explanation.segments
        weights = dict(explanation.local_exp[label])

        # Weighted overlay
        weighted_mask = np.zeros(segments.shape)
        for seg_val, weight in weights.items():
            weighted_mask[segments == seg_val] = weight

        weighted_mask -= weighted_mask.min()
        if weighted_mask.max() != 0:
            weighted_mask /= weighted_mask.max()

        cmap = plt.cm.seismic
        colored_mask = cmap(weighted_mask)[..., :3]
        overlay = 0.5 * img_array + 0.5 * colored_mask

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(img_array)
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        ax[1].imshow(mark_boundaries(overlay, segments))
        ax[1].set_title("LIME Heatmap")
        ax[1].axis('off')

        st.pyplot(fig)

    st.success("LIME explanation complete.")

# Contact Text

st.markdown("""
## 📬 Contact Me
Connect with me on:

<a href="https://github.com/SakshamAnand" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" width="30" />
</a>

<a href="https://linkedin.com/in/saksham-anand05/" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/145/145807.png" width="30" />
</a>

<a href="https://instagram.com/saksham.anand05/" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" width="30" />
</a>
""", unsafe_allow_html=True)

