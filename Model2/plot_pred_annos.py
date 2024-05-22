import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np

# Function to visualize predictions from COCO-format JSON file
def visualize_coco_predictions(json_file):
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    image_info = coco_data['images'][0]
    annotations = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Load the image
    image_path = image_info['file_name']
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']
        segmentation = ann['segmentation']
        score = ann.get('score', None)
        category_name = categories[category_id]

        # Draw bounding box
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(bbox[0], bbox[1] - 5, f'{category_name} ({score:.2f})' if score else category_name, color='red', fontsize=12, weight='bold')

        # Draw segmentation mask
        for seg in segmentation:
            poly = np.array(seg).reshape((len(seg) // 2, 2))
            polygon = Polygon(poly, facecolor='none', edgecolor='blue', linewidth=2)
            ax.add_patch(polygon)

    plt.axis('off')
    plt.show()

# Path to the COCO-format JSON file with predictions
json_file = "predicted_damage_annos.json"

# Visualize the predictions
visualize_coco_predictions(json_file)
