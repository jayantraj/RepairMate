import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the COCO JSON file
with open("Data/val/anns/val_annos.json", "r") as f:
    coco_data = json.load(f)

# Create a dictionary to map image_id to image file paths
image_id_to_filename = {image['id']: image['file_name'] for image in coco_data['images']}

print(image_id_to_filename)

# Create a dictionary to map category_id to category names
category_id_to_name = {category['id']: category['name'] for category in coco_data['categories']}

print(category_id_to_name)

# Function to draw a polygon mask
def draw_polygon(image, segmentation, color):
    pts = np.array(segmentation, np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
    cv2.fillPoly(image, [pts], color=color)

# Function to visualize bounding boxes and masks
def visualize_annotations(image_dir, annotations, image_id_to_filename, category_id_to_name):
    for annotation in annotations:
        image_id = annotation['image_id']
        image_path = os.path.join(image_dir, image_id_to_filename[image_id])
        print(image_path)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Image {image_path} not found.")
            continue
        
        # Draw bounding box
        bbox = annotation['bbox']
        x, y, w, h = map(int, bbox)
        category_id = annotation['category_id']
        category_name = category_id_to_name[category_id]
        color = (0, 255, 0)  # Green for bounding box
        text_color = (255,255,255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

        # Draw mask
        if 'segmentation' in annotation and len(annotation['segmentation']) > 0:
            segmentation = annotation['segmentation'][0]
            mask_color = (255, 0, 0)  # Blue for mask
            draw_polygon(image, segmentation, mask_color)

        # Show the image with annotations
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# Path to the directory containing images
image_dir = "Data/val/imgs"

# Visualize annotations
visualize_annotations(image_dir, coco_data['annotations'], image_id_to_filename, category_id_to_name)
