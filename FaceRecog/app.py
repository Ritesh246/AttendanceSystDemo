import streamlit as st
from deepface import DeepFace
import pandas as pd
import numpy as np
from PIL import Image
import os
import cv2
from datetime import datetime

# === CONFIGURATION ===
KNOWN_PATH = "known_students"  # rename folder to known_students
THRESHOLD = 0.4  # DeepFace verification threshold

st.set_page_config(page_title="Selfie Attendance", layout="centered")
st.title("📸 Selfie-Based Attendance System")
st.markdown("Take a live selfie OR upload an image to mark attendance. Then download your attendance Excel.")

# === Load known faces ===
def load_known_faces():
    students = []
    for file in os.listdir(KNOWN_PATH):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(file)[0]
            path = os.path.join(KNOWN_PATH, file)
            students.append({"name": name, "image": path})
    return students

known_students = load_known_faces()

# === Layout: Upload or Capture Image ===
col1, col2 = st.columns(2)
image_data = None

with col1:
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        image_data = np.array(img.convert("RGB"))
        st.image(img, caption="Uploaded Image", use_column_width=True)

with col2:
    captured_image = st.camera_input("📷 Take Image")
    if captured_image:
        img = Image.open(captured_image)
        image_data = np.array(img.convert("RGB"))
        st.image(img, caption="Captured Selfie", use_column_width=True)

# === Match faces using DeepFace ===
if image_data is not None:
    st.subheader("🔍 Matching Faces...")

    temp_path = "temp_group.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))

    present_students = set()

    for student in known_students:
        try:
            result = DeepFace.verify(
                img1_path=student["image"],
                img2_path=temp_path,
                enforce_detection=False
            )
            if result["verified"] and result["distance"] < THRESHOLD:
                present_students.add(student["name"])
                st.success(f"✅ Marked Present: {student['name']}")
        except Exception as e:
            st.warning(f"Error verifying {student['name']}: {str(e)}")

    # === Define all students and roll numbers ===
    all_students = {
        "Aruna": "A01",
        "Ritesh": "A02",
        "Ishita": "A03",
        "Aai": "A04",
        "Papa": "A05",
        "Kaka": "A06"
        # Add more students here...
    }

    # === Generate attendance sheet ===
    attendance = []
    for name, roll in all_students.items():
        status = "Present" if name in present_students else "Absent"
        attendance.append({"Name": name, "Roll No": roll, "Status": status})

    df = pd.DataFrame(attendance)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"attendance_{timestamp}.xlsx"
    df.to_excel(filename, index=False)

    with open(filename, "rb") as f:
        st.download_button("📁 Download Attendance Excel", f, file_name=filename)

