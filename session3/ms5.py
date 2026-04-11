"""
File: mediapipePose.py
Date: Fall 2025

This program provides a demo showing how to use Mediapipe's body pose detection model, and visualize the results.
"""

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def runPoseDetector(source=0):
    """Sets up the pose landmark model, and runs it on a video feed, visualizing the results"""

    # Set up model
    modelPath = "MediapipeModels/Pose landmark detection/pose_landmarker_full.task"
    base_options = python.BaseOptions(model_asset_path=modelPath)
    options = vision.PoseLandmarkerOptions(base_options=base_options,
                                           output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # Set up camera
    cap = cv2.VideoCapture(source)

    while True:
        gotIm, frame = cap.read()
        if not gotIm:
            break

        # Convert the frame to be a Mediapipe image format
        image = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Run the pose detector on the image
        detect_result = detector.detect(mp_image)

        # TODO: Uncomment this call to run the function that checks if that hands are above the head or not
        res = findHandsUp(detect_result)

        # Visualize the pose skeleton on the frame
        annot_image = visualizeResults(mp_image.numpy_view(), detect_result)
        vis_image = cv2.cvtColor(annot_image, cv2.COLOR_RGB2BGR)

        cv2.putText(vis_image, res, (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)


        # If image segementation was done, display the segmentation masks
        # segMasks = detect_result.segmentation_masks
        # if segMasks is not None and len(segMasks) > 0:
        #     segIm = segMasks[0].numpy_view()
        #     cv2.imshow("SegMask", segIm)
        #     segWrit = 255 * segIm
        #     cv2.imshow("segWrit", segWrit)
        cv2.imshow("Detected", vis_image)

        x = cv2.waitKey(10)
        if x > 0:
            if chr(x) == 'q':
                break
    cap.release()


def visualizeResults(rgb_image, detection_result):
    """
    Draws the pose skeleton on a copy of the input image, based on the data in detection_result
    :param rgb_image: an image in RGB format
    :param detection_result: The results of the pose landmark detector
    :return: a copy of the input image with the pose drawn on it
    """
    annotated_image = np.copy(rgb_image)
    pose_landmarks_list = detection_result.pose_landmarks

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    # Loop through the detected poses to visualize.
    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style)

    return annotated_image


def findHandsUp(detect_result):
    """Takes in the pose landmark information, and for each body detected, determines if the hands are
    above the head or not. Prints a message."""
    # TODO: Look at the hand and head positions and determine whether hands are above head or not
    
    # reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
    # check if hand points (15 / 16) are above nose (0) using y coordinates

    if not detect_result.pose_landmarks: return 'nothing detected'

    for i in range(len(detect_result.pose_landmarks)):
        lms = detect_result.pose_landmarks[i]

        head = lms[0].y 
        left = lms[15].y 
        right = lms[16].y 

        left_up = left < head
        right_up = right < head

        if left_up and right_up: return 'both hands are up'
        elif left_up: return 'left hand is up'
        elif right_up: return 'right hand is up'
        else: return 'both hands are down'


if __name__ == "__main__":
    runPoseDetector(0)
