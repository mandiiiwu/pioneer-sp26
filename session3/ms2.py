"""
hamster_mirror.py
-----------------
Requirements:  pip install opencv-python mediapipe numpy

Put all .JPG files in the same folder as this script.
Models are auto-downloaded on first run.
Press Q to quit.
"""

import ssl
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─── model download ───────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent

MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
}

def download_models():
    # bypass SSL cert verification (same as curl -L) for Mac Python 3.13
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for filename, url in MODEL_URLS.items():
        dest = SCRIPT_DIR / filename
        if not dest.exists():
            print(f"downloading {filename} ...")
            with urllib.request.urlopen(url, context=ctx) as response, \
                 open(dest, "wb") as f:
                f.write(response.read())
            print(f"  saved → {dest}")

# ─── image loading ────────────────────────────────────────────────────────────

IMAGE_FILES = {
    "base":       "base.JPG",
    "nerd":       "nerd.JPG",
    "peacesigns": "peacesigns.JPG",
    "smile1":     "smile1.JPG",
    "smile2":     "smile2.JPG",
    "thumbsdown": "thumbsdown.JPG",
    "thumbsup":   "thumbsup.JPG",
    "tongue1":    "tongue1.JPG",
    "tongue2":    "tongue2.JPG",
}

HAMSTER_SIZE = 320

def load_images():
    images = {}
    for key, fname in IMAGE_FILES.items():
        path = SCRIPT_DIR / fname
        img = cv2.imread(str(path))
        if img is None:
            print(f"[warn] could not load {path}")
            img = np.full((300, 300, 3), 220, dtype=np.uint8)
        images[key] = img
    return images

def resize_hamster(img):
    h, w = img.shape[:2]
    scale = HAMSTER_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((HAMSTER_SIZE, HAMSTER_SIZE, 3), 255, dtype=np.uint8)
    yo = (HAMSTER_SIZE - nh) // 2
    xo = (HAMSTER_SIZE - nw) // 2
    canvas[yo:yo+nh, xo:xo+nw] = resized[:, :, :3]
    return canvas

# ─── hand gesture detection ───────────────────────────────────────────────────

def finger_states(hand_landmarks):
    lm = hand_landmarks
    thumb = np.hypot(lm[4].x - lm[2].x, lm[4].y - lm[2].y) > 0.07
    tip_pip = [(8, 6), (12, 10), (16, 14), (20, 18)]
    others = [lm[t].y < lm[p].y for t, p in tip_pip]
    return [thumb, *others]   # [thumb, index, middle, ring, pinky]

def thumb_direction(hand_landmarks):
    lm = hand_landmarks
    palm_y = lm[9].y
    tip_y  = lm[4].y
    THRESH = 0.08
    if tip_y < palm_y - THRESH: return "up"
    if tip_y > palm_y + THRESH: return "down"
    return "neutral"

def detect_gesture(hand_result):
    if not hand_result.hand_landmarks:
        return None

    peace_seen = False
    for hand_lm in hand_result.hand_landmarks:
        f = finger_states(hand_lm)
        if f[1] and f[2] and not f[3] and not f[4]:
            peace_seen = True
            continue
        if f[0] and not f[1] and not f[2] and not f[3] and not f[4]:
            d = thumb_direction(hand_lm)
            if d == "up":   return "thumbsup"
            if d == "down": return "thumbsdown"
        if not f[0] and f[1] and not f[2] and not f[3] and not f[4]:
            return "nerd"

    return "peacesigns" if peace_seen else None

# ─── facial expression detection ──────────────────────────────────────────────

def mouth_metrics(face_landmarks):
    lm = face_landmarks
    upper, lower = lm[13], lm[14]
    left,  right = lm[61], lm[291]

    mouth_h    = abs(lower.y - upper.y)
    mouth_w    = abs(right.x - left.x) + 1e-6
    open_ratio = mouth_h / mouth_w

    mid_y      = (upper.y + lower.y) / 2
    corner_y   = (left.y + right.y) / 2
    smile_score = mid_y - corner_y   # positive = smiling

    return open_ratio, smile_score

def detect_expression(face_result):
    if not face_result.face_landmarks:
        return None

    # ── tune these ────────────────────────
    OPEN_BIG    = 0.28
    OPEN_SMALL  = 0.12
    SMILE_BIG   = 0.018
    SMILE_SMALL = 0.006
    # ──────────────────────────────────────

    or_, ss = mouth_metrics(face_result.face_landmarks[0])
    if or_ > OPEN_BIG:    return "tongue2"
    if or_ > OPEN_SMALL:  return "tongue1"
    if ss  > SMILE_BIG:   return "smile2"
    if ss  > SMILE_SMALL: return "smile1"
    return None

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    download_models()

    images   = load_images()
    hamsters = {k: resize_hamster(v) for k, v in images.items()}

    # IMAGE mode — no timestamps needed, matches professor's detect() style
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(SCRIPT_DIR / "hand_landmarker.task")
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
    )

    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(SCRIPT_DIR / "face_landmarker.task")
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] could not open webcam")
        sys.exit(1)

    print("hamster mirror running — press Q to quit")

    current_key = "base"
    display_key = "base"
    key_counter = 0
    STABLE_FRAMES = 2

    with (
        mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_det,
        mp_vision.FaceLandmarker.create_from_options(face_opts) as face_det,
    ):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # plain .detect() — no timestamp needed (IMAGE mode)
            hand_result = hand_det.detect(mp_image)
            face_result = face_det.detect(mp_image)

            # pick hamster
            new_key = "base"
            gesture = detect_gesture(hand_result)
            if gesture:
                new_key = gesture
            else:
                expr = detect_expression(face_result)
                if expr:
                    new_key = expr

            # stability filter — avoids one-frame flickers
            if new_key == current_key:
                key_counter += 1
            else:
                key_counter = 0
                current_key = new_key
            if key_counter >= STABLE_FRAMES:
                display_key = current_key

            # draw
            fh, fw = frame.shape[:2]
            frame[0:HAMSTER_SIZE, fw-HAMSTER_SIZE:fw] = hamsters[display_key]

            cv2.putText(frame, display_key,
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 200, 80), 2, cv2.LINE_AA)

            # live debug values for threshold tuning
            if face_result.face_landmarks:
                or_, ss = mouth_metrics(face_result.face_landmarks[0])
                cv2.putText(frame, f"open={or_:.3f}  smile={ss:.4f}",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (180, 180, 180), 1)

            cv2.imshow("hamster mirror", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()