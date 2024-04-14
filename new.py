from flask import Flask, request, jsonify, render_template
import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime

app = Flask(__name__)

# Known face encodings and their names
known_face_encodings = []
known_face_names = []

# Load and encode faces
def load_and_encode_faces():
    base_directory = 'faces/'
    for person_name in os.listdir(base_directory):
        person_directory = os.path.join(base_directory, person_name)
        if os.path.isdir(person_directory):
            for image_file in os.listdir(person_directory):
                image_path = os.path.join(person_directory, image_file)
                if os.path.isfile(image_path):
                    print(f"Encoding {image_path}")
                    image = face_recognition.load_image_file(image_path)
                    try:
                        face_encoding = face_recognition.face_encodings(image)[0]
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(person_name)
                    except IndexError:
                        print(f"No face found in {image_path}, skipping.")

load_and_encode_faces()

# Initialize or update the CSV file for attendance
def initialize_or_update_csv():
    today_column = datetime.now().strftime("%Y-%m-%d")
    if not os.path.isfile('attendance.csv'):
        with open('attendance.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Student Name", today_column])
            for name in set(known_face_names):
                writer.writerow([name, "Absent"])
    else:
        with open('attendance.csv', 'r+', newline='') as file:
            reader = csv.reader(file)
            headers = next(reader)
            if today_column not in headers:
                headers.append(today_column)
                # Collect existing data
                data = list(reader)
                # Append "Absent" for the new column for all rows
                data = [row + ["Absent"] for row in data]
                # Reset file pointer and rewrite the file
                file.seek(0)
                writer = csv.writer(file)
                writer.writerow(headers)
                writer.writerows(data)

def update_attendance(name):
    today_column = datetime.now().strftime("%Y-%m-%d")
    temp_data = []
    updated = False
    with open('attendance.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)
        date_index = headers.index(today_column)
        temp_data.append(headers)
        for row in reader:
            if row[0] == name:
                row[date_index] = "Present"
                updated = True
            temp_data.append(row)
    if updated:
        with open('attendance.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(temp_data)

# Capture photos
def capture_photos(subject_name, num_photos=5):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Photos")

    angles = ["straight", "left", "right", "up", "down"]
    photos_taken = 0

    try:
        # Create a directory for the subject if it doesn't exist
        os.makedirs(f'faces/{subject_name}', exist_ok=True)

        while photos_taken < num_photos:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break
            cv2.imshow("Capture Photos", frame)

            angle = angles[photos_taken % len(angles)]
            instruction = f"Look {angle}. Press space to capture."
            print(instruction)
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                img_name = f"faces/{subject_name}/{subject_name}_{angle}_{photos_taken}.png"
                cv2.imwrite(img_name, frame)
                print(f"{img_name} written!")
                photos_taken += 1

    finally:
        cam.release()
        cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    subject_name = request.form['subject_name']
    num_photos = int(request.form.get('num_photos', 5))
    capture_photos(subject_name, num_photos)
    return jsonify({"message": "Photos captured successfully!"})

@app.route('/recognize', methods=['POST'])
def recognize():
    video_capture = cv2.VideoCapture(0)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        if process_this_frame:
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                face_names.append(name)
                if name != "Unknown":
                    update_attendance(name)
        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Recognition completed!"})

if __name__ == "__main__":
    app.run(debug=True)
