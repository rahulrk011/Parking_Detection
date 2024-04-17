import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_and_pad(image, target_size):
    """Resizes and pads an image to match the target size.

    Args:
        image: The image to be resized and padded.
        target_size: A tuple representing the target width and height.

    Returns:
        A resized and padded image with the same aspect ratio as the original image.
    """
    h, w, _ = image.shape
    target_h, target_w = target_size

    # Calculate scaling factor to maintain aspect ratio
    scale_factor = min(target_w / w, target_h / h)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calculate padding required
    top_pad = (target_h - new_h) // 2
    bottom_pad = target_h - new_h - top_pad
    left_pad = (target_w - new_w) // 2
    right_pad = target_w - new_w - left_pad

    # Pad the image with black pixels
    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded_image

def find_red_rectangles(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Create a mask to isolate red areas
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter rectangular contours
    red_rectangles = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.08 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            red_rectangles.append((x, y, x + w, y + h))

    return red_rectangles

def divide_image(parking_image, base_image):
    # Resize and pad parking image to match base image size
    resized_parking = resize_and_pad(parking_image.copy(), base_image.shape[:2])

    # Find red rectangles in the base image
    rectangles = find_red_rectangles(base_image.copy())

    # Divide parking image based on rectangles
    divided_images = []
    for x, y, x2, y2 in rectangles:
        cropped_parking = resized_parking[y:y2, x:x2]
        divided_images.append(cropped_parking)

    return divided_images

def rgb_to_gray(image):
    # Convert image to NTSC coordinates
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def quadtree_segmentation_dynamic(image, max_light_factor, min_light_factor, max_dark_factor, min_dark_factor):
    h, w = image.shape[:2]

    # Recursive function to decompose the image with dynamic thresholding
    def decompose(image, x, y, width, height):
        block = image[y:y+height, x:x+width]
        mean_intensity = np.mean(block) / 255

        if mean_intensity > 0.5:
            factor = max_light_factor - (max_light_factor - min_light_factor) * mean_intensity
        else:
            factor = max_dark_factor - (max_dark_factor - min_dark_factor) * (1 - mean_intensity)

        if width == 1 or height == 1 or np.max(block) - np.min(block) <= factor:
            # If the block width or height is 1 or the criterion is met, stop dividing and return the block
            return [(x, y, width, height)]

        # If the criterion is not met, divide the block into four quadrants
        new_width = width // 2
        new_height = height // 2
        top_left = decompose(image, x, y, new_width, new_height)
        top_right = decompose(image, x + new_width, y, width - new_width, new_height)
        bottom_left = decompose(image, x, y + new_height, new_width, height - new_height)
        bottom_right = decompose(image, x + new_width, y + new_height, width - new_width, height - new_height)

        # Combine and return the quadrants
        return top_left + top_right + bottom_left + bottom_right

    # Start quadtree decomposition from the top-left corner of the entire image
    return decompose(image, 0, 0, w, h)

# Read the base image
base_image = cv2.imread("base.jpg")

# Read the parking image
parking_image = cv2.imread("parking6.jpg")

# Divide parking image based on red rectangles in base image
divided_images = divide_image(parking_image, base_image)

# Define the maximum and minimum threshold factors for quadtree decomposition
max_light_threshold_factor = 150  # Adjust this value as needed
min_light_threshold_factor = 10   # Adjust this value as needed
max_dark_threshold_factor = 100   # Adjust this value as needed
min_dark_threshold_factor = 5    # Adjust this value as needed

# Define the occupancy threshold factor (adjust this value based on experimentation)
occupancy_threshold_factor = 5  # Adjust this value as needed

# Lists to store data for plotting
light_threshold_factors = [max_light_threshold_factor, min_light_threshold_factor]
dark_threshold_factors = [max_dark_threshold_factor, min_dark_threshold_factor]
segment_percentages = []

# Loop through each divided image and perform quadtree segmentation
for index, divided_image in enumerate(divided_images):
    # Perform quadtree segmentation with dynamic thresholding
    gray_divided_image = rgb_to_gray(divided_image)
    segments = quadtree_segmentation_dynamic(gray_divided_image, max_light_threshold_factor, min_light_threshold_factor,
                                             max_dark_threshold_factor, min_dark_threshold_factor)

    # Calculate the total number of pixels in the image
    total_pixels = gray_divided_image.shape[0] * gray_divided_image.shape[1]

    # Calculate the number of segments generated by quadtree decomposition
    num_segments = len(segments)

    # Calculate the percentage of segments compared to the total number of pixels
    segment_percentage = (num_segments / total_pixels) * 100
    segment_percentages.append(segment_percentage)

    # Determine occupancy status based on the percentage of segments
    if segment_percentage >= occupancy_threshold_factor:
        occupancy_status = "Occupied"
    else:
        occupancy_status = "Vacant"

    print(f"Occupancy Status for Slot {index}.jpg:", occupancy_status)

    # Create a black background
    output_image = np.zeros_like(gray_divided_image)

    # Draw white lines for each segment
    for segment in segments:
        x, y, w, h = segment
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (255), 2)

    # Invert the colors (white lines on black background)
    output_image = cv2.bitwise_not(output_image)

    # Write the resulting image to a file
    cv2.imwrite(f'divided_{index}_quadtree_segmentation_result.jpg', output_image)

# Plotting the threshold factors
plt.figure(figsize=(10, 6))
plt.plot(['Max Light Factor', 'Min Light Factor'], light_threshold_factors, marker='o', label='Light Threshold Factors', color='blue')
plt.plot(['Max Dark Factor', 'Min Dark Factor'], dark_threshold_factors, marker='o', label='Dark Threshold Factors', color='red')
plt.xlabel('Threshold Factor Type')
plt.ylabel('Threshold Factor Value')
plt.title('Threshold Factors for Quadtree Decomposition')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting the occupancy factor
plt.figure(figsize=(6, 4))
plt.bar(range(len(segment_percentages)), segment_percentages, color='green', label='Occupancy Factor')
plt.xlabel('Parking Slot')
plt.ylabel('Segment Percentage (%)')
plt.title('Occupancy Factor for Parking Slots')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
