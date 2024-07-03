import cv2
import numpy as np

def draw_matches(img1, kp1, img2, kp2, matches):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)

    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2

    for m in matches:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
        pt2 = (int(kp2[m.trainIdx].pt[0]) + w1, int(kp2[m.trainIdx].pt[1]))

        cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
        cv2.rectangle(vis, (pt1[0] - 5, pt1[1] - 5), (pt1[0] + 5, pt1[1] + 5), (255, 0, 0), 2)
        cv2.rectangle(vis, (pt2[0] - 5, pt2[1] - 5), (pt2[0] + 5, pt2[1] + 5), (255, 0, 0), 2)

    return vis

def find_object_sift(img1, img2):
    # SIFT
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)

    kp2, des2 = sift.detectAndCompute(img2, None)

    # feature matching
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)

    
    if len(matches) > 100:
        
        vis = draw_matches(img1, kp1, img2, kp2, matches)
        
        cv2.imshow('Matches', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("Not Founded.")

img1 = cv2.imread('eiffel.jpeg')
img2 = cv2.imread('eiffel(1).jpeg')

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

find_object_sift(img1_gray, img2_gray)
