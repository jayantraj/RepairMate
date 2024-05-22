import json
import os
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

def load_coco_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def coco_to_polygon(segmentation):
    """Convert COCO segmentation to Shapely Polygon."""
    if len(segmentation) == 0:
        return None
    poly = Polygon(np.array(segmentation).reshape(-1, 2))
    return poly if poly.is_valid else poly.buffer(0)

def dice_coefficient(poly1, poly2):
    """Compute the Dice coefficient between two polygons."""
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area
    return (2.0 * intersection) / union if union > 0 else 0.0

def convert_square_pixels_to_square_cm(area_in_square_pixels, dpi=96):
    """Convert area from square pixels to square centimeters."""
    pixels_per_cm = dpi / 2.54
    area_in_square_cm = area_in_square_pixels / (pixels_per_cm ** 2)
    return area_in_square_cm

def get_categories(data_json):
    data = load_coco_json(data_json)
    result_dict = {}

    for ann in data['categories']:
        result_dict[ann['id']]=ann['name']
    return result_dict


def main(damage_json, car_parts_json, output_csv, dpi=96):
    # Load annotations
    damage_data = load_coco_json(damage_json)
    car_parts_data = load_coco_json(car_parts_json)
    
    # Create a dictionary to store polygons by image_id
    damage_polygons = {}
    car_part_polygons = {}
    image_id_to_filename = {image['id']: image['file_name'] for image in damage_data['images']}

    # Populate damage polygons
    for ann in damage_data['annotations']:
        image_id = ann['image_id']
        poly = coco_to_polygon(ann['segmentation'][0])
        if poly:
            if image_id not in damage_polygons:
                damage_polygons[image_id] = []
            damage_polygons[image_id].append((ann['category_id'], poly, convert_square_pixels_to_square_cm(poly.area, dpi)))

    # Populate car part polygons
    for ann in car_parts_data['annotations']:
        image_id = ann['image_id']
        poly = coco_to_polygon(ann['segmentation'][0])
        if poly:
            if image_id not in car_part_polygons:
                car_part_polygons[image_id] = []
            car_part_polygons[image_id].append((ann['category_id'], poly, convert_square_pixels_to_square_cm(poly.area, dpi)))

    # Define categories
    car_parts_categories = get_categories(car_parts_json)
    damage_categories = get_categories(damage_json)

    # Average costs of repairing 1 cm^2 area of the part
    # average_costs = {
    #     'headlamp': 0.6,
    #     'rear_bumper': 1.2,
    #     'door': 1650,
    #     'hood': 1250,
    #     'front_bumper': 1250
    # }

    average_costs = {
        'headlamp': 450,
        'rear_bumper': 1250,
        'door': 1650,
        'hood': 1250,
        'front_bumper': 1250,
        'back_bumper': 1250,
        'back_glass':500,
        'back_left_door':1650,
        'back_left_light':450,
        'back_right_door':1650,
        'back_right_light':450,
   
        'front_glass':500,
        'front_left_door':1650,
        'front_left_light':450,
        'front_right_door':1650,
        'front_right_light':450,

        'left_mirror':350,
        'right_mirror':350,
        'tailgate':1000,
        'trunk':2000,
        'wheel':1200 

    }

    # severity multipliers
    severity_multipliers = {
        'minor': 1,
        'moderate': 1.5,
        'severe': 2.0
    }

    # Prepare the dataframe
    columns = ["image_id", "filename"] + [f"{part}_dice" for part in car_parts_categories.values()] + \
              ["unknown", "damage_area_cm2"] + [f"{part}_area_cm2" for part in car_parts_categories.values()] + \
              list(damage_categories.values()) + ["severity_level", "estimated_cost"]
    df = pd.DataFrame(columns=columns)

    # Iterate over each image and compute Dice coefficients and repair costs
    for image_id in damage_polygons:
        if image_id not in car_part_polygons:
            continue

        for damage_id, damage_poly, damage_area in damage_polygons[image_id]:
            row = {col: 0 for col in columns}
            row["image_id"] = image_id
            row["filename"] = image_id_to_filename[image_id]
            row["damage_area_cm2"] = damage_area

            # one damage can have intersection with multiple car parts. We take the sum of the cost of each damaged
            # car part.
            total_estimated_cost = 0
            severity_level = None
     

            for car_part_id, car_part_poly, car_part_area in car_part_polygons[image_id]:

                car_part_name = car_parts_categories.get(car_part_id, "unknown")
                dice = dice_coefficient(damage_poly, car_part_poly)

                row[f"{car_part_name}_dice"] = dice
                # currently we don't use this in the estimated cost formula, it is for future work.
                row[f"{car_part_name}_area_cm2"] = car_part_area

                severity_level = severity_multipliers[damage_categories[damage_id]]

                row["severity_level"] = severity_level

                # %of damage_area*severity_level*average_cost of the part
                total_estimated_cost += dice*severity_level*average_costs[car_part_name]

            row["estimated_cost"] = total_estimated_cost
            row['unknown'] = 1 if all(v == 0 for k, v in row.items() if 'dice' in k) else 0
            row[damage_categories[damage_id]] = 1

            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df['total_cost'] = df.groupby('image_id')['estimated_cost'].transform('sum')
    df['total_cost'] = df['total_cost'].round(2)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    # print(f"CSV file saved to {output_csv}")



if __name__ == "__main__":
    damage_json = "Results/predicted_damage_annos.json"
    car_parts_json = "Results/predicted_car_part_annos.json"
    output_csv = "Results/repair_cost_outputs.csv"
    dpi = 96  # Example DPI, replace with the actual DPI of your images if different
    main(damage_json, car_parts_json, output_csv, dpi)