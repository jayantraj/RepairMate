# This python file is used for building the streamlit app.

# STEP 1: RUN THE CAR PART PREDICTIONS -> It would visualize and generate the output annotations in COCO Format for the Car Parts
# Model1_largedata/car_part_predictions.py

# STEP 2: RUN THE DAMAGE PREDICTIONS -> It would visualize and generate the output annotations in COCO Format for different level of damages. Model2/damage_predictions.py
# STEP 3: RUN The dice_coefficient_repair_cost.py -> to generate the output CSV file 

import streamlit as st
import subprocess
from PIL import Image
import os
import warnings
import sys
import contextlib
import pandas as pd

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release")

# Context manager to suppress stderr
@contextlib.contextmanager
def suppress_stderr():
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr

# Function to run a Python script
def run_script(script_path, *args):
    try:
        with suppress_stderr():
            result = subprocess.run(
                ['python', script_path, *args],
                check=True,
                capture_output=True,
                text=True
            )
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr

# Streamlit app
st.title("RepairMate: Vehicle Damage Detection and Cost Estimation")

# Create an upload directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Upload image
uploaded_file = st.file_uploader("Upload your car image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image
    image_path = os.path.join("uploads", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Running predictions...")

    # Run Script 1 (Car Part Predictions)
    car_part_output_path = "Results/predicted_car_part_annos.json"
    car_part_script = 'Model1_largedata/car_part_predictions.py'
    car_part_class_names = 'Model1_largedata/class_names.txt'
    car_part_model_weights = 'Model1_largedata/model_0009999.pth'
    car_part_output_image_path = "Results/car_part_predictions_visualized.jpg"
    car_part_stdout, car_part_stderr = run_script(car_part_script, image_path, car_part_output_path, car_part_class_names, car_part_model_weights, car_part_output_image_path)
    
    # Display Script 1 outputs
    st.text("Car Part Predictions Output:")
    st.text(car_part_stdout)
    if car_part_stderr:
        st.text("Errors:")
        st.text(car_part_stderr)
    
    # Display Car Part Predictions Image
    if os.path.exists(car_part_output_image_path):
        car_part_image = Image.open(car_part_output_image_path)
        st.image(car_part_image, caption='Car Part Predictions', use_column_width=True)
    
    # Run Script 2 (Damage Predictions)
    damage_output_path = "Results/predicted_damage_annos.json"
    damage_script = 'Model2/damage_predictions.py'
    damage_class_names = 'Model2/class_names.txt'
    damage_model_weights = 'Model2/model_0005999.pth'
    damage_output_image_path = "Results/damage_predictions_visualized.jpg"
    damage_stdout, damage_stderr = run_script(damage_script, image_path, damage_output_path, damage_class_names, damage_model_weights, damage_output_image_path)
    
    # Display Script 2 outputs
    st.text("Damage Predictions Output:")
    st.text(damage_stdout)
    if damage_stderr:
        st.text("Errors:")
        st.text(damage_stderr)
    
    # Display Damage Predictions Image
    if os.path.exists(damage_output_image_path):
        damage_image = Image.open(damage_output_image_path)
        st.image(damage_image, caption='Damage Predictions', use_column_width=True)


    # Run the script to calculate repair costs
    repair_cost_script = 'dice_coefficient_repair_cost.py'
    repair_cost_output_csv = "Results/repair_cost_outputs.csv"
    repair_stdout, repair_stderr = run_script(repair_cost_script, "Results/predicted_damage_annos.json", "Results/predicted_car_part_annos.json", repair_cost_output_csv)
    
    # Display repair cost calculation outputs
    # st.text("Repair Cost Calculation Output:")
    st.text(repair_stdout)
    if repair_stderr:
        # st.text("Errors:")
        # st.text(repair_stderr)
        print('')
    
    # Read and display total cost
    if os.path.exists(repair_cost_output_csv):
        df = pd.read_csv(repair_cost_output_csv)
        if not df.empty and 'total_cost' in df.columns:
            total_cost = df['total_cost'].iloc[0]
            st.write(f"Total Repair Cost: ${total_cost}")
        else:
            # st.write("No Damage Detected...")
            st.write("Please Upload a Alternate/Brighter Image")


