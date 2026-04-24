"""
File: mediapipeHand.py
Date: Fall 2025

This program provides a demo showing how to use Mediapipe's hand pose detection model, and how to visualize
the results.
"""

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def runHandModel(source):
    """Main program, sets up the model, then runs it on a video feed"""

    # Set up model
    modelPath = "MediapipeModels/hand_landmarker.task"
    base_options = python.BaseOptions(model_asset_path=modelPath)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=6)
    detector = vision.HandLandmarker.create_from_options(options)

    # Set up camera
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert camera image to Mediapipe representation
        image = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Run the hand pose detector, receive detection information
        detect_result = detector.detect(mp_image)

        # TODO: Uncomment this to detect whether the hand is open palm or fist
        results = findHandPose(detect_result)

        # Draw the results using mediapipe tools, then display the result
        annot_image = visualizeResults(mp_image.numpy_view(), detect_result)
        vis_image = cv2.cvtColor(annot_image, cv2.COLOR_RGB2BGR)

        for i in range(len(results)):
            cv2.putText(vis_image, results[i], (100, 100*(i+1)),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Detected", vis_image)

        x = cv2.waitKey(10)
        if x > 0:
            if chr(x) == 'q':
                break
    cap.release()


def visualizeResults(rgb_image, detection_result):
    """
    Draws hand skeleton for each hand visible in an image
    :param rgb_image: An RGB image array
    :param detection_result: The results from the hand landmark detector
    :return: a copy of the input array with the hand skeleton drawn on it, labeled with left or right handedness
    """
    annotated_image = np.copy(rgb_image)

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


def findHandPose(detect_results):
    """Takes in the hand position results and determines whether the hand is an open palm, fingers up,
    or a closed fist"""
    # TODO: for each detected hand, extract the appropriate features and check them. Print the result
    if not detect_results.hand_landmarks: return ['no hands open']

    # based off: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    pips = [6, 10, 14, 18]
    dips = [7, 11, 15, 19]

    returns = []

    for i in range(len(detect_results.hand_landmarks)):
        lms = detect_results.hand_landmarks[i]
        hand = detect_results.handedness[i][0].category_name 

        count = 0 

        for j in range(len(pips)):
            dip = lms[dips[j]].y
            pip = lms[pips[j]].y

            if dip < pip: count += 1

        if count == 4: pose = 'open palm'
        elif count == 0: pose = 'closed fist'
        else: pose = f'{count} fingers up'

        returns.append(f'{hand} hand: {pose}')
    
    return returns
    


if __name__ == "__main__":
    runHandModel(0)
