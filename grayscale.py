################
## convert images to grayscale
################

import cv2
import os

# Define the folder path
input_folder = os.path.expanduser("./source")
output_folder = os.path.expanduser("./Pictures_Grayscale")

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image file in the folder
for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder, filename)
    
    # Check if it's an image file
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img = cv2.imread(filepath)
        if img is not None:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, gray_img)
            print(f"Saved grayscale image: {output_path}")

print("Processing complete!")

################
## resize and fill in the blank with 0
################

import cv2
import os
import numpy as np

# Define the folder paths
input_folder = "images"
output_folder = "images/resized"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image file in the folder
for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder, filename)
    
    # Check if it's an image file
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # Get new dimensions (20% smaller)
            h, w = img.shape
            new_w, new_h = int(w * 1.38), int(h * 1.38)

            # Resize the image
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # # Create a blank canvas with original dimensions
            # final_img = np.zeros((h, w), dtype=np.uint8)

            # # Place resized image at the top-left corner
            # final_img[:new_h, :new_w] = resized_img

            # Save the processed image
            output_path = os.path.join(output_folder, filename)
            # cv2.imwrite(output_path, final_img)
            cv2.imwrite(output_path, resized_img)
            print(f"Saved resized image: {output_path}")

print("Processing complete!")


################
## template matched images
################

import cv2
import os
import numpy as np

# Define paths
image_folder = "source"
template_path = "template.jpg"
output_folder = "matched"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the template image
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
th, tw = template.shape

# Process each image in the folder
for filename in os.listdir(image_folder):
    filepath = os.path.join(image_folder, filename)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            # Perform template matching
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # If match confidence is high, save the image
            if max_val > 0.8:  # Adjust this threshold if needed
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, img)
                print(f"Template matched! Saved: {output_path}")

print("Processing complete!")



#####################
import cv2
import os

# Define input and output directories
input_folder = "./images"
output_folder = "./images/cropped"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define cropping dimensions
crop_x = 1766 - 1280  # Crop start X position (right-bottom area)
crop_y = 993 - 720  # Crop start Y position

# Process images
for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder, filename)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img = cv2.imread(filepath)

        if img is not None and img.shape[:2] == (993, 1766):
            # Crop the right-bottom area
            cropped_img = img[crop_y:993, crop_x:1766]

            # Save the cropped image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_img)
            print



########################
import cv2
import os
import numpy as np

# Define paths
image_folder = "."
template_path = "template.jpg"
output_folder = "matched"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the template image
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
th, tw = template.shape

# Process each image in the folder

img = cv2.imread('defect_1.jpg', cv2.IMREAD_GRAYSCALE)

if img is not None:
    # Perform template matching
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # If match confidence is high, save the image

    cv2.imwrite('output.jpg', img)
    print(f"Template matched! Saved")

print("Processing complete!")