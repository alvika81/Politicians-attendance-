import streamlit as st
import face_recognition
import cv2
import numpy as np
import sqlite3
from datetime import datetime
from PIL import Image
import os

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS attendance (name TEXT, time TEXT)")
    conn.commit()
    conn.close()

def mark_attendance(name):
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("INSERT INTO attendance VALUES (?, ?)", (name, str(datetime.now())))
    conn.commit()
    conn.close()

init_db()

# ---------------- LOAD FACES ----------------
known_face_encodings = []
known_face_names = []

face_folder = "known_faces"
for filename in os.listdir(face_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = face_recognition.load_image_file(os.path.join(face_folder, filename))
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(filename.split(".")[0])  # name from file

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Politician Attendance", layout="wide")
st.title("🗳️ Politician Face Recognition Attendance System")

st.write("Take a photo using your camera. The system will detect if the face matches a known politician and mark attendance.")

uploaded_file = st.camera_input("Take a picture")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img)

    rgb_frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    name_found = "Unknown"

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name_found = known_face_names[best_match_index]
            mark_attendance(name_found)

    st.subheader("Result:")
    if name_found != "Unknown":
        st.success(f"✅ {name_found} detected. Attendance marked.")
    else:
        st.error("❌ Face not recognized.")

# ---------------- SHOW ATTENDANCE ----------------
if st.button("Show Attendance Log"):
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT * FROM attendance")
    rows = c.fetchall()
    conn.close()
    if rows:
        for row in rows:
            st.write(f"📌 {row[0]} at {row[1]}")
    else:
        st.write("No attendance records yet.")
