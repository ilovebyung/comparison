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

def match_image(image, template):
    h, w = template.shape[::]
    result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = min_loc  # Change to max_loc for all except for TM_SQDIFF
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # Crop the image
    matched_area = image[top_left[1]:bottom_right[1],
                         top_left[0]:bottom_right[0]]
    return matched_area

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

def check_difference(input_image, reconstructed_image):
    # Ensure inputs are NumPy arrays
    if not isinstance(input_image, np.ndarray) or not isinstance(reconstructed_image, np.ndarray):
        raise TypeError("Both input and reconstructed images must be NumPy arrays.")

    # Check if images have the same shape
    if input_image.shape != reconstructed_image.shape:
        raise ValueError("Input and reconstructed images must have the same shape.")

    # Normalize input image if values are in [0, 1] range
    if input_image.max() <= 1.0:
        input_image = (input_image * 255).astype(np.uint8)
    else:
        input_image = input_image.astype(np.uint8)

    # Normalize reconstructed image if values are in [0, 1] range
    if reconstructed_image.max() <= 1.0:
        reconstructed_image = (reconstructed_image * 255).astype(np.uint8)
    else:
        reconstructed_image = reconstructed_image.astype(np.uint8)

    # Compute absolute difference
    image_a = cv2.subtract(input_image, reconstructed_image)
    image_b = cv2.subtract(reconstructed_image, input_image)
    difference = cv2.absdiff(image_a, image_b)
    # difference = cv2.absdiff(input_image, reconstructed_image)
    # difference = cv2.applyColorMap(difference, cv2.COLORMAP_MAGMA)
    return difference

def extract_blue(input_image):
    # Load the color image
    image = cv2.imread(input_image)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the blue color range in HSV
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])

    # Create a binary mask where blue colors are white and the rest are black
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Convert the original image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the mask to the grayscale image
    blue_area_gray = cv2.bitwise_and(gray, gray, mask=mask)
    return blue_area_gray

if __name__ == '__main__':
    ## You can change the directory names here if needed
    convert_images_to_grayscale(input_dir='Pictures', output_dir='Pictures_Grayscale')

    ## matched images
    template = cv2.imread('/home/byungsoo/Documents/comparison/template.jpg', 0)
    plt.imshow(template, cmap='gray')

    os.chdir('/home/byungsoo/Documents/comparison/Pictures_Grayscale')
    files = os.listdir('/home/byungsoo/Documents/comparison/Pictures_Grayscale')

    for file in files:
        if file.endswith('.jpg'):
            image = cv2.imread(file, 0)
            # result = match_and_align_image(image, template)
            result = match_image(image, template)
            cv2.imwrite(f'/home/byungsoo/Documents/comparison/Pictures_Matched/{file}', result)
            # cv2.imwrite(f'/home/byungsoo/Documents/comparison/Pictures_Matched/backup/{file}', result)
            
    plt.imshow(result, cmap='gray')

    image = extract_blue('blue.png')
    plt.imshow(image, cmap='gray')




    
