import mediapipe as mp
import cv2 as cv
import re
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

outFile = open('./Josh/HandData.csv', 'w')
outFile.write("Time,Hand,")

for i in range(21):
  outFile.write("Node " + str(i + 1))
  if i != 20:
    outFile.write(",")
outFile.write("\n")

video_path = input("[Optional - ENT for webcam] Input full path of video to operate on: ")

if video_path == "":
  cap = cv.VideoCapture(0)
  writer = None
else:
  cap = cv.VideoCapture(video_path)
  print(f"Opening {video_path}...")
  if not cap.isOpened():
    print("Error opening video file. Please try again.")
  width, height = cap.read()[1].shape
  writer = cv.VideoWriter("./Videos/Edited/TestVideo.mp4v", cv.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))

with mp_hands.Hands(
    model_complexity = 0,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    imageHeight, imageWidth, _ = image.shape
    outStr = ""
    handStr = ""
    multihand_tracker = 0
    iteration = 0

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Determine right or left hand
        for hand in results.multi_handedness:
          # The way mediapipe detects hands is flipped, so we flip it back
          if hand.classification[0].label == "Right":
            handStr += "Left"
          elif hand.classification[0].label == "Left":
            handStr += "Right"

          if handStr == "RightLeft" and iteration == 0:
            multihand_tracker = 1
            iteration = 1
          elif handStr == "LeftRight" and iteration == 0:
            multihand_tracker = 3
            iteration = 1
          elif re.match("(Left){2,}", handStr) != None:
            handStr = "Left"
          elif re.match("(Right){2,}", handStr) != None:
            handStr = "Right"            

        outStr = ""
        # Coordinates of the hand landmarks
        for point in mp_hands.HandLandmark:
          normalized = hand_landmarks.landmark[point]
          outStr += f"({normalized.x * imageWidth}:{normalized.y * imageHeight}:{normalized.z})"

        # Timestamp
        dt = datetime.now()
        dt_str = dt.strftime("%d-%m-%Y @ %H:%M:%S")

        outFile.write(dt_str + ",")

        # Handling for two hands
        if multihand_tracker == 1:
          outFile.write("Right,")
          multihand_tracker = 2
        elif multihand_tracker == 2:
          outFile.write("Left,")
          multihand_tracker = 0
        elif multihand_tracker == 3:
          outFile.write("Left,")
          multihand_tracker = 4
        elif multihand_tracker == 4:
          outFile.write("Right,")
          multihand_tracker = 0
        else:
          outFile.write(handStr + ",")

        # Adjust commas in points due to .csv format
        # outStr = re.sub(",", ":", outStr)

        # Add commas for .csv format
        outStr = re.sub("\)\(", "),(", outStr)

        # Write output
        outFile.write(outStr + "\n")

        if writer != None:
          writer.write(image)

    # Flip the image horizontally for a selfie-view display.
    cv.imshow('MediaPipe Hands', cv.flip(image, 1))

    # Quit
    if cv.waitKey(5) & 0xFF == 27:
      outFile.close()
      break

cap.release()
if writer != None:
  writer.release()