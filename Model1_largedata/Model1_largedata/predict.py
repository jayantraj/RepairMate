import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
from detectron2.structures import Boxes, Instances

# Function to setup configuration and create predictor
def setup_cfg(model_weights, score_threshold,num_classes):
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
def visualize_predictions(image, outputs, metadata):
    v = Visualizer(image[:, :, ::-1],
                   metadata=metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW)   # remove the colors of unsegmented pixels
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return v.get_image()[:, :, ::-1]

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

# Main function
def main():

    # Path to model weights
    model_weights = "model_0009999.pth"
    
    class_list_file = 'class_names.txt'

    # List of classes used in training
    with open(class_list_file,'r') as reader:
        thing_classes = [l[:-1] for l in reader.readlines()]
    
    # Register dataset with the correct metadata
    MetadataCatalog.get("my_dataset").thing_classes = thing_classes

    score_threshold=0.5
    
    # Setup configuration and create predictor
    cfg = setup_cfg(model_weights, score_threshold, num_classes=len(thing_classes))
    predictor = DefaultPredictor(cfg)
    
    # Load Metadata
    metadata = MetadataCatalog.get("my_dataset")
    
    # Path to input image
    image_path = "Data/val/78.jpg"

    
    # Make predictions
    image, outputs = predict(image_path, predictor)
    
    # Extract prediction details
    pred_boxes, pred_classes, pred_masks, pred_scores = extract_prediction_details(outputs)

    # print(pred_boxes)
    # print(pred_classes)
    # print(pred_masks)
    # print(pred_scores)
    
    # Filter predictions
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
    vis_image = visualize_predictions(image, outputs, metadata)
    
    # Display the image
    plt.figure(figsize=(14, 10))
    plt.imshow(vis_image)
    plt.axis('off')
    plt.show()

    return image,outputs,metadata
    
    # Optionally save the visualization result
    # output_path = "path/to/save/output/image.jpg"
    # cv2.imwrite(output_path, vis_image)

if __name__ == "__main__":
    main()
