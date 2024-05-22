import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.structures import Boxes, Instances
import json
import sys

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release")

"""
This file makes the prediction for the damage category and write the result in COCO-Format to the JSON File.

"""

# Function to setup configuration and create predictor
def setup_cfg(model_weights, score_threshold, num_classes):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = 'cpu'
    return cfg

# Function to make predictions
def predict(image_path, predictor):
    im = cv2.imread(image_path)
    outputs = predictor(im)
    return im, outputs

# Function to visualize the predictions
def visualize_predictions(image, outputs, metadata,save_path=None):
    v = Visualizer(image[:, :, ::-1],
                   metadata=metadata, 
                   scale=0.5)  # Keeping the original colors of the image
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_image = v.get_image()[:, :, ::-1]
    if save_path:
        plt.figure(figsize=(14, 10))
        plt.imshow(result_image)
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
    return result_image

# Function to extract and print prediction details
def extract_prediction_details(outputs):
    instances = outputs["instances"]
    pred_boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    pred_classes = instances.pred_classes if instances.has("pred_classes") else None
    pred_masks = instances.pred_masks if instances.has("pred_masks") else None
    pred_scores = instances.scores if instances.has("scores") else None
    
    return pred_boxes, pred_classes, pred_masks, pred_scores

# Function to filter predictions to keep only the highest confidence prediction per class
def filter_predictions(pred_boxes, pred_classes, pred_masks, pred_scores):
    unique_classes = pred_classes.unique()
    filtered_boxes = []
    filtered_classes = []
    filtered_masks = []
    filtered_scores = []

    for cls in unique_classes:
        class_indices = (pred_classes == cls).nonzero(as_tuple=True)[0]
        class_scores = pred_scores[class_indices]
        max_score_index = class_indices[class_scores.argmax()]

        filtered_boxes.append(pred_boxes.tensor[max_score_index])
        filtered_classes.append(pred_classes[max_score_index])
        filtered_masks.append(pred_masks[max_score_index])
        filtered_scores.append(pred_scores[max_score_index])
    
    if len(filtered_boxes) == 0:
        return Boxes(torch.empty(0, 4)), torch.empty(0, dtype=torch.int64), torch.empty(0, pred_masks.shape[1], pred_masks.shape[2], dtype=torch.uint8), torch.empty(0, dtype=torch.float32)
    
    filtered_boxes = Boxes(torch.stack(filtered_boxes))
    filtered_classes = torch.tensor(filtered_classes, dtype=torch.int64)
    filtered_masks = torch.stack(filtered_masks)
    filtered_scores = torch.tensor(filtered_scores, dtype=torch.float32)

    return filtered_boxes, filtered_classes, filtered_masks, filtered_scores

# Function to convert binary mask to COCO RLE format
def binary_mask_to_polygon(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:  # Contour should have at least 5 points to form a polygon
            segmentation.append(contour)
    return segmentation

# Function to save predictions to COCO format JSON file
def save_to_coco_format(image_id, image_path, pred_boxes, pred_classes, pred_masks, pred_scores, output_json):
    annotations = []
    for i in range(len(pred_boxes)):
        box = pred_boxes[i].tensor.numpy().tolist()
        bbox = [box[0][0], box[0][1], box[0][2] - box[0][0], box[0][3] - box[0][1]]
        mask = pred_masks[i].numpy().astype(np.uint8)
        segmentation = binary_mask_to_polygon(mask)
        annotation = {
            "image_id": image_id,
            "category_id": pred_classes[i].item(),
            "bbox": bbox,
            "segmentation": segmentation,
            "score": pred_scores[i].item(),
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "id": i
        }
        annotations.append(annotation)
    
    coco_output = {
        "images": [
            {
                "id": image_id,
                "file_name": image_path,
                "height": pred_masks.shape[1],
                "width": pred_masks.shape[2]
            }
        ],
        "annotations": annotations,
        "categories": [
            {"id": i, "name": name} for i, name in enumerate(MetadataCatalog.get("my_dataset").thing_classes)
        ]
    }
    
    with open(output_json, 'w') as f:
        json.dump(coco_output, f, indent=4)


"""
The input is uploaded image, output directory, class_names.txt, model_weights, output_image_path which comes from app.py.
"""

# Main function
def main(image_path,output_json,class_list_file,model_weights,output_image_path):
    # Path to model weights
    # model_weights = "model_0005999.pth"
    
    # class_list_file = 'class_names.txt'

    # List of classes used in training
    with open(class_list_file, 'r') as reader:
        thing_classes = [l.strip() for l in reader.readlines()]
    
    # Register dataset with the correct metadata
    MetadataCatalog.get("my_dataset").thing_classes = thing_classes

    score_threshold = 0.5
    
    # Setup configuration and create predictor
    cfg = setup_cfg(model_weights, score_threshold, num_classes=len(thing_classes))
    predictor = DefaultPredictor(cfg)
    
    # Load Metadata
    metadata = MetadataCatalog.get("my_dataset")
    
    # Path to input image
    # image_path = "Data/test/imgs/test_image_5.jpg"
    
    # Make predictions
    image, outputs = predict(image_path, predictor)
    
    # Extract prediction details
    pred_boxes, pred_classes, pred_masks, pred_scores = extract_prediction_details(outputs)
    
    # Filter predictions: picking the highest confidence bounding box for each class
    pred_boxes, pred_classes, pred_masks, pred_scores = filter_predictions(pred_boxes, pred_classes, pred_masks, pred_scores)
    
    # Create new Instances object with filtered predictions
    filtered_instances = Instances(image.shape[:2])
    filtered_instances.pred_boxes = pred_boxes
    filtered_instances.pred_classes = pred_classes
    filtered_instances.pred_masks = pred_masks
    filtered_instances.scores = pred_scores
    
    # Update the outputs with filtered instances
    outputs["instances"] = filtered_instances
    
    # Visualize predictions
    # vis_image = visualize_predictions(image, outputs, metadata)

    visualize_predictions(image, outputs, metadata, save_path=output_image_path)
    
    # Display the image :  Plt.show() doesn't work in streamlit app. So, we are saving the visualization
    # plt.figure(figsize=(14, 10))
    # plt.imshow(vis_image)
    # plt.axis('off')
    # plt.show()
    
    # Save the predictions to COCO format JSON file
    # output_json = "predicted_damage_annos.json"
    save_to_coco_format(image_id=1, image_path=image_path, pred_boxes=pred_boxes, pred_classes=pred_classes, pred_masks=pred_masks, pred_scores=pred_scores, output_json=output_json)

if __name__=="__main__":
    image_path = sys.argv[1]
    output_json = sys.argv[2]
    class_names_list = sys.argv[3]
    model_weights = sys.argv[4]
    output_image_path = sys.argv[5]
    main(image_path, output_json,class_names_list,model_weights,output_image_path)
