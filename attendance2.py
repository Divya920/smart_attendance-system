from deepface import DeepFace
import cv2
import os
from datetime import datetime
import pandas as pd
import time

# Paths
DB_PATH = 'capture_images'
ATTENDANCE_FILE = 'attendance.csv'
SIMILARITY_THRESHOLD = 0.4  # Lower = stricter match
TEMP_IMAGE = "temp.jpg"
DELAY_SECONDS = 3  # Delay before marking attendance

# Create attendance file if not exists
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=['Name', 'Time', 'Count']).to_csv(ATTENDANCE_FILE, index=False)

def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    # Count previous occurrences
    previous_count = df[df['Name'] == name].shape[0]
    current_count = previous_count + 1  # Increment count
    current_time = datetime.now().strftime("%H:%M:%S")
    # Append new entry
    df = pd.concat([df, pd.DataFrame([[name, current_time, current_count]], columns=['Name', 'Time', 'Count'])], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)
    return current_count

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Camera started... Press 'q' to exit")

detected_name = None
detection_time = None
attendance_marked_flag = False
stop_system = False  # Stop after marking attendance

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite(TEMP_IMAGE, small_frame)

    try:
        result = DeepFace.find(img_path=TEMP_IMAGE, db_path=DB_PATH, enforce_detection=False)

        if len(result) > 0 and len(result[0]) > 0:
            identity_path = result[0].iloc[0]['identity']
            name = os.path.splitext(os.path.basename(identity_path))[0]

            similarity = result[0].iloc[0]['VGG-Face_cosine'] if 'VGG-Face_cosine' in result[0].columns else 0

            if similarity < SIMILARITY_THRESHOLD:
                # New face detected
                if detected_name != name:
                    detected_name = name
                    detection_time = time.time()
                    attendance_marked_flag = False

                # Display the name immediately
                cv2.putText(frame, f"Name: {detected_name}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                elapsed_time = time.time() - detection_time

                # During waiting period
                if not attendance_marked_flag and elapsed_time < DELAY_SECONDS:
                    cv2.putText(frame, "Marking attendance...", (50, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                # After delay, mark attendance and show count
                elif not attendance_marked_flag and elapsed_time >= DELAY_SECONDS:
                    count = mark_attendance(detected_name)
                    cv2.putText(frame, f"Your attendance has been marked  {count}", (50, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    stop_system = True
                # If already marked, keep showing confirmation
                elif attendance_marked_flag:
                    cv2.putText(frame, f"Your attendance has been marked ✅ (Count: {count})", (50, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                cv2.putText(frame, "Unknown face", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                detected_name = None

    except Exception as e:
        cv2.putText(frame, "No face detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        detected_name = None

    cv2.imshow('Smart Attendance System', frame)

    if stop_system:
        cv2.waitKey(2000)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Attendance process ended ✅")
