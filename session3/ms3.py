"""
File: mediapipeFaceLandmark.py
Date: Fall 2025

This program provides a demo showing how to use Mediapipe's facial landmark model, and to visualize the results.
"""

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import matplotlib.pyplot as plt


def runFacialLandmarks(source=0):
    # Set up model
    modelPath = "MediapipeModels/face_landmarker_v2_with_blendshapes.task"
    base_options = python.BaseOptions(model_asset_path=modelPath)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # Set up camera
    cap = cv2.VideoCapture(source)

    while True:
        gotIm, frame = cap.read()
        if not gotIm:
            break

        # Convert image to Mediapipe image format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Run facial landmark detector
        detect_result = detector.detect(mp_image)

        # TODO: Uncomment this call to run the function that checks whether eyes are open or closed
        res, s1, s2 = findEyes(detect_result)

        annot_image = visualizeResults(mp_image.numpy_view(), detect_result)
        vis_image = cv2.cvtColor(annot_image, cv2.COLOR_RGB2BGR)
        cv2.putText(vis_image, res, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.putText(vis_image, f'left eye: {s1:.3f}, right eye: {s2:.3f}', (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Detected", vis_image)

        x = cv2.waitKey(10)
        if x > 0:
            if chr(x) == 'q':
                break
            if chr(x) == 'b' and len(detect_result.face_landmarks) > 0:
                plot_face_blendshapes_bar_graph(detect_result.face_blendshapes[0])
    cap.release()


def visualizeResults(rgb_image, detection_result):
    """
    Draw the face landmark mesh onto a copy of the input RGB image and returns it
    :param rgb_image: an image in RGB format (as a Numpy array)
    :param detection_result: The results of running the face landmarker model
    :return: a copy of rgb_image with face landmark mesh drawn on it
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    """
    Creates a plt bar graph to show how much each blendshape is present in a given image
    :param face_blendshapes: output from the blendshapes model
    :return:
    """
    # Extract the face blendshapes category names and scores.
    for face_blsh in face_blendshapes:
        print(face_blsh)
        print(face_blsh.category_name, face_blsh.score)


    face_blsh_names = [face_blsh_category.category_name for face_blsh_category in face_blendshapes]
    face_blsh_scores = [face_blsh_category.score for face_blsh_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blsh_ranks = range(len(face_blsh_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blsh_ranks, face_blsh_scores, label=[str(x) for x in face_blsh_ranks])
    ax.set_yticks(face_blsh_ranks, face_blsh_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blsh_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def findEyes(detect_result):
    """Takes in the facial landmark results and determines, for each face located, whether the
        eyes are open or closed. Print a message"""
    # TODO: Look at the blendshapes for the eyes and determine if the eyes are open or closed

    if not detect_result.face_blendshapes: return 'no face detected', 0, 0

    blshs = detect_result.face_blendshapes[0]
    left_score = 0 
    right_score = 0

    for face_blsh in blshs:
        if face_blsh.category_name == 'eyeBlinkLeft': left_score = face_blsh.score 
        if face_blsh.category_name == 'eyeBlinkRight': right_score = face_blsh.score
    
    l_closed = left_score >= 0.5
    r_closed = right_score >= 0.5

    if l_closed and r_closed: return 'both eyes are closed', left_score, right_score
    elif l_closed: return 'only left eye closed', left_score, right_score
    elif r_closed: return 'only right eye closed', left_score, right_score
    else: return 'both eyes are open', left_score, right_score


if __name__ == "__main__":
    runFacialLandmarks(0)
