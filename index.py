import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
 
video_capture = cv2.VideoCapture(0)
 
jobs_image = face_recognition.load_image_file("photos/jobs.png")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]
 
aryan_image = face_recognition.load_image_file("photos/aryan.png")
aryan_encoding = face_recognition.face_encodings(aryan_image)[0]
 
shridhar_image = face_recognition.load_image_file("photos/shridhar.png")
shridhar_encoding = face_recognition.face_encodings(shridhar_image)[0]
 
tesla_image = face_recognition.load_image_file("photos/tesla.png")
tesla_encoding = face_recognition.face_encodings(tesla_image)[0]

Devansh_image = face_recognition.load_image_file("photos/Devansh.png")
Devansh_encoding = face_recognition.face_encodings(Devansh_image)[0]
 
known_face_encoding = [
jobs_encoding,
#ratan_tata_encoding,
shridhar_encoding,
tesla_encoding,
Devansh_encoding,
aryan_encoding,
]
 
known_faces_names = [
"JOBS",
#"RATAN TATA",
"Shridhar",
"TESLA",
"Devansh",
"Aryan",
]
 
students = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True
 
 
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
 
 
 
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)
 
while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding, tolerance=0.5)
            name="Unknown"
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
 
            face_names.append(name)
            if name != "Unknown":
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
 
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H hours : %M minutes : %S seconds")
                    print(f"Writing attendance for {name} at {current_time}")
                    lnwriter.writerow([name,current_time])
                    f.flush()
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (0,0,255)  # Red color for non-registered faces
                thickness              = 3
                lineType               = 2

                cv2.putText(frame,'Face not recognized', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
f.close()