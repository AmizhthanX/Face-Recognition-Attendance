import cv2
import numpy as np
import os
import face_recognition
import csv
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

# CSV File Paths
STUDENTS_CSV = "students.csv"
ATTENDANCE_CSV = "attendance.csv"

# Initialize CSV Files
def setup_csv():
    if not os.path.exists(STUDENTS_CSV):
        with open(STUDENTS_CSV, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Roll No", "Face Encoding"])
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Roll No", "Name", "Date", "Time"])

# Register Student and Capture Face
def register_student(name, roll_no):
    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Instructions", "Look at the camera to capture your face")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        cv2.imshow("Capturing Face - Press 'c' to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            if face_locations:
                img_path = f"faces/{roll_no}.jpg"
                if not os.path.exists("faces"):
                    os.makedirs("faces")
                cv2.imwrite(img_path, frame)
                cap.release()
                cv2.destroyAllWindows()
                
                img = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(img)
                if len(encodings) > 0:
                    encoding = encodings[0].tolist()
                    with open(STUDENTS_CSV, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([name, roll_no, encoding])
                    messagebox.showinfo("Success", "Student Registered Successfully")
                    return
                else:
                    messagebox.showerror("Error", "No face detected. Try again.")
                    return
    cap.release()
    cv2.destroyAllWindows()

# Recognize and Mark Attendance Once Per Day
def mark_attendance():
    known_encodings = []
    known_rolls = []
    known_names = []
    marked_today = set()
    
    with open(STUDENTS_CSV, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if row[2]:  # Ensure encoding exists
                known_names.append(row[0])
                known_rolls.append(row[1])
                known_encodings.append(eval(row[2]))
    
    # Load today's attendance
    today_date = str(datetime.now().date())
    with open(ATTENDANCE_CSV, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if row[2] == today_date:
                marked_today.add(row[1])
    
    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Start Attendance", "Click OK and wait for face detection")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding) if len(known_encodings) > 0 else []
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
            name = "Unknown"
            
            if best_match_index is not None and matches[best_match_index]:
                roll_no = known_rolls[best_match_index]
                name = known_names[best_match_index]
                date = datetime.now().date()
                time = datetime.now().time()
                
                if roll_no not in marked_today:
                    with open(ATTENDANCE_CSV, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([roll_no, name, date, time])
                    
                    update_attendance_list()
                    messagebox.showinfo("Attendance Marked", f"{name} ({roll_no}) marked present.")
                    marked_today.add(roll_no)
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition - Press q to exit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Update GUI Attendance List
def update_attendance_list():
    attendance_tree.delete(*attendance_tree.get_children())
    with open(ATTENDANCE_CSV, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            attendance_tree.insert("", "end", values=row)

# GUI Application
def main():
    setup_csv()
    global attendance_tree
    
    root = tk.Tk()
    root.title("Face Recognition Attendance System")
    root.geometry("800x500")
    
    tk.Label(root, text="Student Name").pack()
    name_entry = tk.Entry(root)
    name_entry.pack()
    
    tk.Label(root, text="Roll Number").pack()
    roll_entry = tk.Entry(root)
    roll_entry.pack()
    
    tk.Button(root, text="Register", command=lambda: register_student(name_entry.get(), roll_entry.get())).pack()
    tk.Button(root, text="Mark Attendance", command=mark_attendance).pack()
    
    tk.Label(root, text="Attendance Records", font=("Arial", 12, "bold")).pack()
    columns = ("Roll No", "Name", "Date", "Time")
    attendance_tree = ttk.Treeview(root, columns=columns, show='headings')
    for col in columns:
        attendance_tree.heading(col, text=col)
    attendance_tree.pack(fill=tk.BOTH, expand=True)
    
    update_attendance_list()
    
    root.mainloop()

if __name__ == "__main__":
    main()
