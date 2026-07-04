import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time

from predict import predict_cloud_mask

st.set_page_config(
    page_title="CloudFree Bharat",
    page_icon="🛰️",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🛰️ CloudFree Bharat")

use_case = st.sidebar.selectbox(
    "Select Application",
    [
        "Agriculture Monitoring",
        "Flood Assessment",
        "Forest Monitoring",
        "Urban Planning"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
AI-Assisted Satellite Intelligence Platform

Bharat Antariksh Hackathon Prototype
"""
)

# =========================
# HEADER
# =========================

st.title("🛰️ CloudFree Bharat")

st.subheader(
    "AI-Assisted Satellite Cloud Removal & Land Intelligence Platform"
)

st.write(
    """
Upload a cloud-covered satellite image and generate a cloud-free
visualization with satellite intelligence analytics.
"""
)

uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PROCESS
# =========================

if uploaded_file is not None:

    start_time = time.time()

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Cloud Detection

    mask = predict_cloud_mask(image_np)

    # Reconstruction

    result = cv2.inpaint(
        image_np,
        mask,
        7,
        cv2.INPAINT_TELEA
    )

    # =========================
    # LAND ANALYTICS
    # =========================

    hsv = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2HSV
    )

    vegetation_mask = cv2.inRange(
        hsv,
        np.array([35, 40, 40]),
        np.array([90, 255, 255])
    )

    water_mask = cv2.inRange(
        hsv,
        np.array([90, 50, 50]),
        np.array([140, 255, 255])
    )

    total_pixels = (
        result.shape[0] *
        result.shape[1]
    )

    vegetation_pixels = np.count_nonzero(
        vegetation_mask
    )

    water_pixels = np.count_nonzero(
        water_mask
    )

    cloud_pixels = np.count_nonzero(
        mask
    )

    vegetation_percent = (
        vegetation_pixels / total_pixels
    ) * 100

    water_percent = (
        water_pixels / total_pixels
    ) * 100

    cloud_percent = (
        cloud_pixels / total_pixels
    ) * 100

    land_percent = max(
        0,
        100 -
        vegetation_percent -
        water_percent
    )

    end_time = time.time()

    processing_time = round(
        end_time - start_time,
        2
    )

    # =========================
    # METRICS
    # =========================

    st.subheader(
        "📊 Satellite Intelligence Dashboard"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "☁️ Cloud %",
        f"{cloud_percent:.2f}"
    )

    c2.metric(
        "🌳 Vegetation %",
        f"{vegetation_percent:.2f}"
    )

    c3.metric(
        "🌊 Water %",
        f"{water_percent:.2f}"
    )

    c4.metric(
        "🌍 Land %",
        f"{land_percent:.2f}"
    )

    st.info(
        f"⚡ Processing Time: {processing_time} seconds"
    )

    # =========================
    # AI SUMMARY
    # =========================

    st.divider()

    st.subheader(
        "🧠 AI Analysis Summary"
    )

    summary = []

    if cloud_percent > 20:
        summary.append(
            "High cloud contamination detected."
        )
    else:
        summary.append(
            "Moderate cloud contamination detected."
        )

    if vegetation_percent > 30:
        summary.append(
            "Dense vegetation cover identified."
        )

    if water_percent > 10:
        summary.append(
            "Water bodies detected."
        )

    summary.append(
        f"Recommended application: {use_case}"
    )

    for item in summary:
        st.write("•", item)

    # =========================
    # IMAGES
    # =========================

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image_np,
            caption="Original Satellite Image",
            use_container_width=True
        )

    with col2:
        st.image(
            result,
            caption="Cloud-Free Reconstruction",
            use_container_width=True
        )

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.image(
            mask,
            caption="Cloud Detection Mask",
            use_container_width=True
        )

    with col4:
        st.image(
            vegetation_mask,
            caption="Vegetation Detection",
            use_container_width=True
        )

    # =========================
    # APPLICATIONS
    # =========================

    st.divider()

    st.subheader(
        "🌍 Potential Applications"
    )

    a, b = st.columns(2)

    with a:
        st.success(
            "🌾 Agriculture Monitoring"
        )

        st.success(
            "🌳 Forest Assessment"
        )

    with b:
        st.success(
            "🌊 Flood Mapping"
        )

        st.success(
            "🏙 Urban Planning"
        )

    # =========================
    # DOWNLOAD IMAGE
    # =========================

    result_pil = Image.fromarray(
        result
    )

    img_buffer = io.BytesIO()

    result_pil.save(
        img_buffer,
        format="JPEG"
    )

    st.download_button(
        label="📥 Download Reconstructed Image",
        data=img_buffer.getvalue(),
        file_name="cloudfree_bharat_output.jpg",
        mime="image/jpeg"
    )

    # =========================
    # REPORT
    # =========================

    report = f"""
CloudFree Bharat Analysis Report

Cloud Coverage: {cloud_percent:.2f} %

Vegetation Cover: {vegetation_percent:.2f} %

Water Bodies: {water_percent:.2f} %

Land Area: {land_percent:.2f} %

Processing Time: {processing_time} sec

Application Mode: {use_case}
"""

    st.download_button(
        label="📄 Download Analysis Report",
        data=report,
        file_name="analysis_report.txt",
        mime="text/plain"
    )

    # =========================
    # FUTURE SCOPE
    # =========================

    st.divider()

    st.subheader(
        "🚀 Future Scope"
    )

    st.markdown("""
- LISS-IV Satellite Integration
- Sentinel-1 SAR Fusion
- Deep Learning Cloud Segmentation
- Generative AI Reconstruction
- Real-Time Monitoring Dashboard
""")

    st.success(
        "✅ Analysis Complete"
    )