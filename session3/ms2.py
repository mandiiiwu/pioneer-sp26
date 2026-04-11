"""
File: mediapipeFaceDetect.py
Date: Fall 2025

This program provides a demo showing how to use Mediapipe's simple face detection model, and to visualize the results.
"""
import math
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
CIRCLE_COLOR = (0, 255, 0)  # green
TEXT_COLOR = (0, 255, 255)  # cyan, remembering that this is applied to an RGB, not a BGR, image


def runFaceDetect(source=0):
    """Main program, sets up the blaze face detection model and then runs it on a video feed."""

    # Set up model
    modelPath = "MediapipeModels/blaze_face_short_range.tflite"
    # base_options = python.BaseOptions(model_asset_path=modelPath)
    # options = vision.FaceDetectorOptions(base_options=base_options)
    # detector = vision.FaceDetector.create_from_options(options)
    base_options = python.BaseOptions(model_asset_path=modelPath)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    # Set up camera
    cap = cv2.VideoCapture(source)

    while True:
        gotIm, frame = cap.read()
        if not gotIm:
            break

        # added image mirroring so more accurate 
        frame = cv2.flip(frame, 1)

        # Convert the frame to a Mediapipe image representation
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Run the face detector model on the image
        detect_result = detector.detect(mp_image)

        # TODO: Uncomment this call to run the function that checks which way each detected face is pointing
        orient, diff = findFacing(detect_result, mp_image.numpy_view())

        annot_image = visualizeResults(mp_image.numpy_view(), detect_result)

        # Display the results on screen
        vis_image = cv2.cvtColor(annot_image, cv2.COLOR_RGB2BGR)
        cv2.putText(vis_image, f'{orient}, diff: {diff}', (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Detected", vis_image)

        x = cv2.waitKey(10)
        if x > 0:
            if chr(x) == 'q':
                break
    cap.release()


def _normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value):
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not is_valid_normalized_value(normalized_x):
        normalized_x = max(0.0, min(1.0, normalized_x))
    if not is_valid_normalized_value(normalized_y):
        normalized_y = max(0.0, min(1.0, normalized_y))
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualizeResults(image, detection_result):
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualized.
    Returns: Image with bounding boxes.
    """

    # Copy the original image and make changes to the copy
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box for each face detected
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        kps = {'left eye': (255, 0, 0), # red 
               'right eye': (0, 0, 255), # blue 
               'nose': (0, 255, 0), # green 
               'mouth': (0, 255, 255), # teal? 
               'left ear': (255, 0, 255), # purple 
               'right ear': (255, 255, 0) # brown 
        }

        for i in range(len(detection.keypoints)):
            kp = detection.keypoints[i] 
            x, y = _normalized_to_pixel_coordinates(kp.x, kp.y, width, height)
            cv2.circle(annotated_image, (x, y), 3, list(kps.values())[i], -1)
            cv2.putText(annotated_image, list(kps.keys())[i], (x-50, y+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, list(kps.values())[i])

        # # Draw face keypoints for each face detected
        # for keypoint in detection.keypoints:
        #     keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
        #     cv2.circle(annotated_image, keypoint_px, 3, CIRCLE_COLOR, -1)

        # Draw category label and confidence score as text on bounding box
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image


def findFacing(detect_results, image):
    """Takes in the face detection results and determines, for each face located, whether the
    face is pointing forward, to the left, or to the right. It prints a message with the results."""
    # TODO: for each face detected, determine the facing from the relative positions of each eye and
    # TODO: the edge of the face on that side

    height, width, _ = image.shape

    # ALGORITHM:
    # checked for horiz. dists btwn left eye to left ear, 
    # then right eye to right ear
    # then made into ratios (dist1/dist2 + dist2/dist1)
    # after many rounds of experimentation, found that
    # there the difference btwn the ratios could determine 
    # facial orientation 

    if detect_results is not None:
        for detection in detect_results.detections:
            l_eye = detection.keypoints[0]
            l_ear = detection.keypoints[4] 

            r_eye = detection.keypoints[1] 
            r_ear = detection.keypoints[5]

            kps = [l_eye, l_ear, r_eye, r_ear]
            coords = []
            for kp in kps:
                coords.append(_normalized_to_pixel_coordinates(kp.x, kp.y, width, height))
            
            x1, y1 = coords[0] 
            x2, y2 = coords[1]
            dist1 = abs(x2-x1)

            x3, y3 = coords[2]
            x4, y4 = coords[3]
            dist2 = abs(x4-x3)

            ratio1 = dist1 / dist2 if dist2 != 0 else 999999999 
            ratio2 = dist2 / dist1 if dist1 != 0 else 999999999
            diff = ratio2 - ratio1 
            # print(f'ratio 1: {ratio1}\nratio 2: {ratio2}\ndiff: {diff}')

            if abs(diff) <= 1.5: return 'front facing', diff 
            elif diff > 1.5: return 'facing left', diff 
            else: return 'facing right', diff 
    return 'no face detected', 0




if __name__ == "__main__":
    runFaceDetect(0)
