import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def template_matching(image, template):
    h, w = template.shape[::]
    result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = min_loc  # Change to max_loc for all except for TM_SQDIFF
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # Crop the image
    matched_area = image[top_left[1]:bottom_right[1],
                         top_left[0]:bottom_right[0]]
    return matched_area
    # cropped_area = matched_area[200:800, 400:1900]
    # return cropped_area

def align_images(image, template):

    # Detect ORB features
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image, None)
    kp2, des2 = orb.detectAndCompute(template, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Select good matches
    good_matches = matches[:20]  # Adjust number of good matches as needed

    # Get source and destination points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the input image
    aligned_image = cv2.warpPerspective(image, M, (template.shape[1], template.shape[0]))

    return aligned_image



if __name__ == '__main__':
    template = cv2.imread('template.jpg', 0)
    image = cv2.imread('sample.jpg', 0)

    plt.imshow(template, cmap='gray')
    plt.imshow(image, cmap='gray')

    # matched_area has the same shape
    matched_area = template_matching(image, template)
    plt.imshow(matched_area, cmap='gray')

    # matched_area has the same shape
    aligned_image = align_images(matched_area, template)
    plt.imshow(aligned_image, cmap='gray')





