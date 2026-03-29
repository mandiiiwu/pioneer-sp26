# a lot of this is copied almost directly from camshift code segments in SampleCode. 

import numpy as np
import cv2
import os

showBackProj = False
showHistMask = False
frame = None
hist = None

def show_hist(hist):
    bin_count = hist.shape[0]
    bin_w = 24
    img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
    for i in range(bin_count):
        h = int(hist[i])
        cv2.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h),
                      (int(180.0 * i / bin_count), 255, 255), -1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow('hist', img)


cam = cv2.VideoCapture(0)
ret, frame = cam.read()

if frame is not None:
    (hgt, wid, dep) = frame.shape
    cv2.namedWindow('camshift')
    cv2.namedWindow('hist')
    cv2.moveWindow('hist', 700, 100)

    track_window = (0, 0, wid, hgt)

    # read in the cropped image of the object instead of hardcoding colors
    histImage = cv2.imread(f'{os.getcwd()}/session2/object.png')
    histImage_hsv = cv2.cvtColor(histImage, cv2.COLOR_BGR2HSV)
    maskedHistIm = cv2.inRange(histImage_hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    hist = cv2.calcHist([histImage_hsv], [0], maskedHistIm, [16], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    hist = hist.reshape(-1)
    show_hist(hist)

    while True:
        ret, frame = cam.read()
        vis = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        if showHistMask:
            vis[mask == 0] = 0

        prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        prob &= mask
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        track_box, track_window = cv2.CamShift(prob, track_window, term_crit)

        # if box is way too elongated and center isn't on the obj,
        # it's probably stretched across 2 objs -> reset window to re-lock on one
        (cx, cy), (bw, bh), _ = track_box
        if bw > 0 and bh > 0 and max(bw, bh) / min(bw, bh) > 3.0:
            cx, cy = int(cx), int(cy)

            # sample mask at ellipse center
            # if 0, its too dark -> center on bg and not object 

            # restrict cx/cy so we dont go out of bounds on mask array 

            if mask[max(0, min(cy, hgt-1)), max(0, min(cx, wid-1))] == 0:

                # shorter size as approx size of obj (min 40 px)
                # so theres enough pixels for camshift to actually converge 
                sz = max(int(min(bw, bh)), 40)

                # build small square window w (cx, cy) 
                # restricted to frame bounds 
                x0 = max(0, cx - sz // 2)
                y0 = max(0, cy - sz // 2)
                window = (x0, y0, min(wid, x0+sz) - x0, min(hgt, y0+sz) - y0)

                # rerun camshift w/ new window n force it to pick one obj 
                box, window = cv2.CamShift(prob, window, term_crit)

        if showBackProj:
            vis[:] = prob[..., np.newaxis]
        try:
            cv2.ellipse(vis, track_box, (0, 0, 255), 2)
        except:
            print("Track box:", track_box)

        cv2.imshow('camshift', vis)

        ch = chr(0xFF & cv2.waitKey(5))
        if ch == 'q':
            break
        elif ch == 'b':
            showBackProj = not showBackProj
        elif ch == 'v':
            showHistMask = not showHistMask

cv2.destroyAllWindows()