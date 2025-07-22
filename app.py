import streamlit as st
from PIL import Image
import numpy as np
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import cv2

# === CONFIGURATION ===
KNOWN_PATH = "known_faces"
THRESHOLD = 0.4  # lower = stricter match

# === Streamlit UI ===
st.title("📸 Selfie-Based Attendance System")
st.markdown("Take a live selfie OR upload an image to mark attendance. Then download your attendance Excel.")

# === Load known faces ===
@st.cache_resource
def load_known_faces():
    known = []
    for file in os.listdir(KNOWN_PATH):
        if file.lower().endswith(("jpg", "jpeg", "png")):
            path = os.path.join(KNOWN_PATH, file)
            known.append({
                "name": os.path.splitext(file)[0],
                "path": path
            })
    return known

known_faces = load_known_faces()

# === UI Layout ===
col1, col2 = st.columns(2)
image_data = None

with col1:
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        image_data = np.array(img)

with col2:
    captured_image = st.camera_input("📷 Take Image")
    if captured_image:
        img = Image.open(captured_image).convert("RGB")
        st.image(img, caption="Captured Selfie", use_column_width=True)
        image_data = np.array(img)

# === Face Matching Logic ===
if image_data is not None:
    st.subheader("🔍 Processing...")
    matched_names = set()

    try:
        # Detect faces in the uploaded/captured image
        faces = DeepFace.extract_faces(img_path=image_data, enforce_detection=False)

        for face in faces:
            face_img = face["face"]
            # Save the detected face temporarily
            cv2.imwrite("temp_face.jpg", cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

            # Compare with each known face
            for known in known_faces:
                result = DeepFace.verify(
                    img1_path="temp_face.jpg",
                    img2_path=known["path"],
                    enforce_detection=False,
                    model_name='VGG-Face',
                    distance_metric='cosine'
                )

                if result["verified"] and result["distance"] < THRESHOLD:
                    matched_names.add(known["name"])
                    st.success(f"✅ Marked Present: {known['name']}")
                    break  # stop checking once matched

        os.remove("temp_face.jpg")  # Clean up temp file

    except Exception as e:
        st.error(f"Error: {e}")

    # === Full Class List ===
    all_students = {
        "Aruna": "A01",
        "Ritesh": "A02",
        "Ishita": "A03",
        "Aai": "A04",
        # Add more if needed
    }

    attendance = []
    for name, roll in all_students.items():
        status = "Present" if name in matched_names else "Absent"
        attendance.append({"Name": name, "Roll No": roll, "Status": status})

    df = pd.DataFrame(attendance)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"attendance_{timestamp}.xlsx"
    df.to_excel(filename, index=False)

    with open(filename, "rb") as f:
        st.download_button("📁 Download Attendance Excel", f, file_name=filename)
