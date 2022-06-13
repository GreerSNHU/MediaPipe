import mediapipe as mp
import cv2 as cv
import re
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

outFile = open(f'./Trackers/Data/Face{datetime.now().strftime("%d-%m-%y_%H-%M")}.csv', 'w')
outFile.write("Time,")

for i in range(478):
  outFile.write("Node " + str(i + 1))
  if i != 477:
    outFile.write(",")
outFile.write("\n")

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    imageHeight, imageWidth, _ = image.shape
    outStr = ""

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image = image,
            landmark_list = face_landmarks,
            connections = mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image = image,
            landmark_list = face_landmarks,
            connections = mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image = image,
            landmark_list = face_landmarks,
            connections = mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
        
        for landmark in face_landmarks.landmark:
          outStr += f"({landmark.x * imageWidth}:{landmark.y * imageHeight}:{landmark.z})"
        
        dt = datetime.now()
        dt_str = dt.strftime("%d-%m-%Y @ %H:%M:%S")

        outFile.write(dt_str + ",")
        # Add commas for .csv format
        outStr = re.sub("\)\(", "),(", outStr)

        outFile.write(outStr + "\n")
        
    # Flip the image horizontally for a selfie-view display.
    cv.imshow('MediaPipe Face Mesh', cv.flip(image, 1))
    if cv.waitKey(5) & 0xFF == 27:
      break
cap.release()