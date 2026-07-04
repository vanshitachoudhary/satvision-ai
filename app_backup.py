import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time

from predict import predict_cloud_mask

st.set_page_config(
    page_title="CloudFree Bharat",
    layout="wide"
)

st.title("🛰️ CloudFree Bharat")
st.subheader("AI-Assisted Satellite Cloud Removal & Land Intelligence Platform")

st.write("""
Upload a cloud-covered satellite image and generate a reconstructed
cloud-free visualization along with land analytics.
""")

uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

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

    # --------------------------------
    # LAND ANALYTICS
    # --------------------------------

    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)

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

    total_pixels = result.shape[0] * result.shape[1]

    vegetation_pixels = np.count_nonzero(vegetation_mask)
    water_pixels = np.count_nonzero(water_mask)

    vegetation_percent = (
        vegetation_pixels / total_pixels
    ) * 100

    water_percent = (
        water_pixels / total_pixels
    ) * 100

    land_percent = max(
        0,
        100 - vegetation_percent - water_percent
    )

    # --------------------------------
    # CLOUD ANALYTICS
    # --------------------------------

    cloud_pixels = np.count_nonzero(mask)

    cloud_percent = (
        cloud_pixels / total_pixels
    ) * 100

    end_time = time.time()

    processing_time = round(
        end_time - start_time,
        2
    )

    # --------------------------------
    # METRICS
    # --------------------------------

    st.subheader("📊 Satellite Analytics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Cloud Coverage %",
        f"{cloud_percent:.2f}"
    )

    c2.metric(
        "Vegetation %",
        f"{vegetation_percent:.2f}"
    )

    c3.metric(
        "Water %",
        f"{water_percent:.2f}"
    )

    c4.metric(
        "Land %",
        f"{land_percent:.2f}"
    )

    st.info(
        f"Processing Time: {processing_time} seconds"
    )

    st.divider()

    # --------------------------------
    # IMAGES
    # --------------------------------

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
            caption="Detected Cloud Mask",
            use_container_width=True
        )

    with col4:
        st.image(
            vegetation_mask,
            caption="Vegetation Detection",
            use_container_width=True
        )

    # --------------------------------
    # DOWNLOAD IMAGE
    # --------------------------------

    result_pil = Image.fromarray(result)

    image_buffer = io.BytesIO()

    result_pil.save(
        image_buffer,
        format="JPEG"
    )

    st.download_button(
        label="📥 Download Reconstructed Image",
        data=image_buffer.getvalue(),
        file_name="cloudfree_bharat_output.jpg",
        mime="image/jpeg"
    )

    # --------------------------------
    # REPORT
    # --------------------------------

    report = f"""
CloudFree Bharat Analysis Report

Cloud Coverage: {cloud_percent:.2f} %

Vegetation Cover: {vegetation_percent:.2f} %

Water Bodies: {water_percent:.2f} %

Land Area: {land_percent:.2f} %

Processing Time: {processing_time} sec
"""

    st.download_button(
        label="📄 Download Analysis Report",
        data=report,
        file_name="analysis_report.txt",
        mime="text/plain"
    )

    st.success("✅ Analysis Complete")