import mediapipe as mp
import cv2 as cv
import re
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
outFile = open('./Josh/LandmarkData.csv', 'w')
outFile.write("Time,Hand,")

for i in range(21):
  outFile.write("Node " + str(i + 1))
  if i != 20:
    outFile.write(",")
outFile.write("\n")

# For webcam input:
cap = cv.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
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
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    imageHeight, imageWidth, _ = image.shape
    outStr = ""

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
            outStr += "Left-"
          elif hand.classification[0].label == "Left":
            outStr += "Right-"

        # Coordinates of the hand landmarks
        for point in mp_hands.HandLandmark:
          normalized = hand_landmarks.landmark[point]
          pixelCoords = mp_drawing._normalized_to_pixel_coordinates(normalized.x, normalized.y, imageHeight, imageWidth)
          outStr += str(pixelCoords)

        # Adjust commas in points due to .csv format
        outStr = re.sub(",", " :", outStr)

        # Add commas for .csv format
        outStr = re.sub("\)\(", "),(", outStr)
        outStr = re.sub("-", ",", outStr)

        # Timestamp
        dt = datetime.now()
        dt_str = dt.strftime("%d-%m-%Y @ %H:%M:%S")

        # Write output
        outFile.write(dt_str + "," + outStr + "\n")

    # Flip the image horizontally for a selfie-view display.
    cv.imshow('MediaPipe Hands', cv.flip(image, 1))

    # Quit
    if cv.waitKey(5) & 0xFF == 27:
      outFile.close()
      break
cap.release()
