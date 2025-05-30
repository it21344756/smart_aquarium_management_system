import json
import os
import cv2

def find_image_and_get_dimensions(image_name, search_directory):
    """
    Searches for an image by name in the given directory and returns its dimensions.

    :param image_name: Name of the image to search for (e.g., 'image.jpg').
    :param search_directory: Directory to search for the image.
    :return: Tuple of (width, height) if found, otherwise None.
    """
    for root, _, files in os.walk(search_directory):
        for file in files:
            if file == image_name:
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                if image is not None:
                    height, width = image.shape[:2]
                    return width, height
                else:
                    print(f"Error: Unable to load image at {image_path}")
                    return None
    print(f"Image '{image_name}' not found in '{search_directory}'")
    return None


# Directory paths
search_dir = "C:\\Users\\Hashan\\Desktop\\test\\dataset\\val\\images"
via_json_path = "C:\\Users\\Hashan\\Desktop\\test\\dataset\\val\\test.json"  # Replace with your file path
output_dir = "C:\\Users\\Hashan\\Desktop\\test\\dataset\\val\\label"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the VIA JSON file
with open(via_json_path, "r") as f:
    via_data = json.load(f)

# Define class mapping
class_mapping = {"disease_detected": 0}  # Update if there are more classes

# Process each image in the VIA JSON
for image_id, image_data in via_data.items():
    # Get image name and corresponding dimensions
    image_name = f"{image_data['filename']}"  # Ensure correct filename from JSON
    dimensions = find_image_and_get_dimensions(image_name, search_dir)
    if not dimensions:
        continue  # Skip if image is not found or cannot be loaded
    image_width, image_height = dimensions

    # Prepare output file
    output_file = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")

    with open(output_file, "w") as out_file:
        regions = image_data.get("regions", [])
        for region in regions:
            # Get class index
            class_name = region["region_attributes"]["names"]
            class_id = class_mapping.get(class_name, -1)  # Default to -1 if class not found
            if class_id == -1:
                continue  # Skip unknown classes

            # Extract polygon points
            points_x = region["shape_attributes"]["all_points_x"]
            points_y = region["shape_attributes"]["all_points_y"]

            # Normalize points
            normalized_points = []
            for x, y in zip(points_x, points_y):
                normalized_x = x / image_width
                normalized_y = y / image_height
                normalized_points.append(normalized_x)
                normalized_points.append(normalized_y)

            # Format the output as: <class-index> <x1> <y1> <x2> <y2> ...
            points_str = " ".join(f"{p:.6f}" for p in normalized_points)
            out_file.write(f"{class_id} {points_str}\n")

print(f"Annotations saved to {os.path.abspath(output_dir)}")
