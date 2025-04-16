import os
from PIL import Image, UnidentifiedImageError
import cv2
import matplotlib.pyplot as plt
import numpy as np

def convert_images_to_grayscale(input_dir='Pictures', output_dir='Pictures_Grayscale'):
    """
    Reads images from an input directory, converts them to grayscale,
    and saves them to an output directory.

    Args:
        input_dir (str): The path to the directory containing the original images.
                         Defaults to 'Pictures'.
        output_dir (str): The path to the directory where grayscale images
                          will be saved. Defaults to 'Pictures_Grayscale'.
    """
    # --- 1. Setup Directories ---
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define absolute paths for input and output directories relative to the script
    abs_input_dir = os.path.join(script_dir, input_dir)
    abs_output_dir = os.path.join(script_dir, output_dir)

    # Check if the input directory exists
    if not os.path.isdir(abs_input_dir):
        print(f"Error: Input directory '{abs_input_dir}' not found.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(abs_output_dir):
        try:
            os.makedirs(abs_output_dir)
            print(f"Created output directory: '{abs_output_dir}'")
        except OSError as e:
            print(f"Error creating output directory '{abs_output_dir}': {e}")
            return

    print(f"Input directory: '{abs_input_dir}'")
    print(f"Output directory: '{abs_output_dir}'")

    # --- 2. Process Images ---
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # List all files in the input directory
    try:
        files = os.listdir(abs_input_dir)
    except OSError as e:
        print(f"Error listing files in input directory '{abs_input_dir}': {e}")
        return

    # Define common image file extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')

    for filename in files:
        # Construct the full path to the file
        input_filepath = os.path.join(abs_input_dir, filename)

        # Check if it's a file and has a valid image extension (case-insensitive)
        if os.path.isfile(input_filepath) and filename.lower().endswith(image_extensions):
            try:
                # Open the image
                with Image.open(input_filepath) as img:
                    # Convert the image to grayscale ('L' mode)
                    grayscale_img = img.convert('L')

                    # Construct the output filename and path
                    output_filename = f"grayscale_{filename}"
                    output_filepath = os.path.join(abs_output_dir, output_filename)

                    # Save the grayscale image
                    grayscale_img.save(output_filepath)
                    print(f"Converted and saved: '{output_filepath}'")
                    processed_count += 1

            except UnidentifiedImageError:
                print(f"Skipping: '{filename}' - Cannot identify image file (possibly corrupted or not an image).")
                skipped_count += 1
            except IOError as e:
                print(f"Error processing file '{filename}': {e}")
                error_count += 1
            except Exception as e:
                print(f"An unexpected error occurred processing '{filename}': {e}")
                error_count += 1
        elif os.path.isfile(input_filepath):
            # Optional: Log files that are skipped because they are not images
            # print(f"Skipping non-image file: '{filename}'")
            skipped_count += 1

    # --- 3. Summary ---
    print("\n--- Processing Summary ---")
    print(f"Total files processed: {processed_count}")
    print(f"Total files skipped (non-image or errors): {skipped_count + error_count}")
    if error_count > 0:
        print(f"  - Errors encountered: {error_count}")
    print("--------------------------")

def match_and_align_image(image, template):
    # Template matching
    h, w = template.shape[::]
    result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    matched_area = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    # Image alignment
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(matched_area, None)
    kp2, des2 = orb.detectAndCompute(template, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:20]
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned_image = cv2.warpPerspective(matched_area, M, (template.shape[1], template.shape[0]))
    
    return aligned_image

if __name__ == '__main__':
    # You can change the directory names here if needed
    convert_images_to_grayscale(input_dir='Pictures', output_dir='Pictures_Grayscale')

    # match and align images
    template = cv2.imread('template.jpg', 0)
    image = cv2.imread('sample.jpg', 0)
    
    plt.imshow(template, cmap='gray')
    plt.imshow(image, cmap='gray')
    
    result = match_and_align_image(image, template)
    plt.imshow(result, cmap='gray')