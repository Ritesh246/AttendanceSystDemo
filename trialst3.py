import streamlit as st  # 👉 Streamlit used for UI
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os
from PIL import Image

# === CONFIGURATION ===
KNOWN_PATH = "known_faces"
TOLERANCE = 0.5

# === Title Heading ===
st.title("📸 Selfie-Based Attendance System")
st.markdown("Take a live selfie OR upload an image to mark attendance. Then download your attendance Excel.")

# === Load Known Faces ===
@st.cache_resource
def load_known_faces():
    known_encodings = []
    known_names = []
    for file in os.listdir(KNOWN_PATH):
        img_path = os.path.join(KNOWN_PATH, file)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])
    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

# === Layout: Side-by-side buttons ===
col1, col2 = st.columns(2)
image_data = None

# 👉 Upload image button
with col1:
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        image_data = np.array(img.convert("RGB"))  # ✅ convert PIL to RGB np.array
        st.image(img, caption="Uploaded Image", use_column_width=True)


# 👉 Webcam capture button
with col2:
    captured_image = st.camera_input("📷 Take Image")
    if captured_image:
        img = Image.open(captured_image)
        image_data = np.array(img.convert("RGB"))  # ✅ convert PIL to RGB np.array
        st.image(img, caption="Captured Selfie", use_column_width=True)


# === Process and Generate Attendance ===
if image_data is not None:
    st.subheader("🔍 Processing...")

    face_locations = face_recognition.face_locations(image_data)
    face_encodings = face_recognition.face_encodings(image_data, face_locations)
    marked_attendance = set()

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=TOLERANCE)
        for i, matched in enumerate(matches):
            if matched:
                marked_attendance.add(known_names[i])
                st.success(f"✅ Marked Present: {known_names[i]}")

    # === Full Student List ===
    all_students = {
        "Aruna": "A01",
        "Ritesh": "A02",
        "Ishita": "A03",
        "Aai": "A04",
        "Papa": "A05",
        "Kaka": "A06"
        # Add more students here...
    }

    # === Prepare Excel Data ===
    attendance = []
    for name, roll in all_students.items():
        status = "Present" if name in marked_attendance else "Absent"
        attendance.append({"Name": name, "Roll No": roll, "Status": status})

    df = pd.DataFrame(attendance)

    # === Save and Offer Excel Download ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"attendance_{timestamp}.xlsx"
    df.to_excel(filename, index=False)

    with open(filename, "rb") as f:
        st.download_button("📁 Download Attendance Excel", f, file_name=filename)

