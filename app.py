import streamlit as st
import subprocess
from PIL import Image
import os
import warnings
import sys
import contextlib
import pandas as pd
import json

# Suppress Intel MKL warnings
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

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

# Function to filter out specific Intel MKL warnings
def filter_warnings(stderr):
    filtered_lines = [
        line for line in stderr.split('\n')
        if not line.startswith("Intel MKL WARNING")
    ]
    return '\n'.join(filtered_lines)

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
        stderr_filtered = filter_warnings(result.stderr)
        return result.stdout, stderr_filtered
    except subprocess.CalledProcessError as e:
        stderr_filtered = filter_warnings(e.stderr)
        return e.stdout, stderr_filtered

# Function to save prediction data
def save_prediction_data(image_name, car_part_output, damage_output, repair_cost_output):
    if not os.path.exists("user_data"):
        os.makedirs("user_data")
    data = {
        "car_part_output": car_part_output,
        "damage_output": damage_output,
        "repair_cost_output": repair_cost_output,
        "car_part_image_path": "Results/car_part_predictions_visualized.jpg",
        "damage_image_path": "Results/damage_predictions_visualized.jpg"
    }
    base_name = os.path.splitext(image_name)[0]
    with open(f"user_data/{base_name}.json", "w") as f:
        json.dump(data, f)

# Function to load prediction data
def load_prediction_data(image_name):
    base_name = os.path.splitext(image_name)[0]
    file_path = f"user_data/{base_name}.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        st.error(f"Data for {image_name} not found.")
        return None

# Function to clear current predictions display
def clear_predictions():
    st.session_state.clear()

# Streamlit app
st.set_page_config(page_title="RepairMate", layout="wide")
st.title("RepairMate: Vehicle Damage Detection and Cost Estimation")

# Add custom CSS for additional styling
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #ff7f0e !important;  /* Vibrant orange color for buttons */
        color: white !important;
        border-radius: 5px !important;
    }
    .stTextInput > div > input {
        background-color: #2ca02c !important;  /* Vibrant green color for text boxes */
        color: white !important;
        border-radius: 5px !important;
    }
    .stImage {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create directories if they don't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")
if not os.path.exists("Results"):
    os.makedirs("Results")

# Initialize session state for uploaded file
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = ""

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Section")
    uploaded_file = st.file_uploader("Upload your car image...", type=["jpg", "jpeg", "png"], key="file_uploader")

# User history section
with st.sidebar:
    st.header("User History")
    user_history = [f for f in os.listdir("uploads") if os.path.isfile(os.path.join("uploads", f))]
    user_history = [""] + user_history  # Add a blank option
    selected_image = st.selectbox("Select an image to view predictions", user_history, key="selected_image")

if selected_image and selected_image != "":
    # Clear the current user-uploaded image
    st.session_state.uploaded_file = None
    clear_predictions()
    st.subheader(f"Predictions for {selected_image}")
    data = load_prediction_data(selected_image)
    if data:
        with st.expander("Car Part Predictions Output"):
            st.text(data.get("car_part_output", "No data available"))
            car_part_image_path = data.get("car_part_image_path")
            if car_part_image_path and os.path.exists(car_part_image_path):
                car_part_image = Image.open(car_part_image_path)
                st.image(car_part_image, caption='Car Part Predictions', use_column_width=True)

        with st.expander("Damage Predictions Output"):
            st.text(data.get("damage_output", "No data available"))
            damage_image_path = data.get("damage_image_path")
            if damage_image_path and os.path.exists(damage_image_path):
                damage_image = Image.open(damage_image_path)
                st.image(damage_image, caption='Damage Predictions', use_column_width=True)

        with st.expander("Repair Cost Calculation Output"):
            st.text(data.get("repair_cost_output", "No data available"))
            repair_cost_output_csv = "Results/repair_cost_outputs.csv"
            if os.path.exists(repair_cost_output_csv):
                df = pd.read_csv(repair_cost_output_csv)
                total_cost = df['total_cost'].iloc[0]
                st.subheader(f"Approximate Repair Cost: ${total_cost}")

elif uploaded_file is not None:
    # Clear previous predictions display
    clear_predictions()
    st.session_state.uploaded_file = uploaded_file
    
    # Reset selected image to blank
    st.session_state.selected_image = ""

    # Save the uploaded image
    image_path = os.path.join("uploads", uploaded_file.name)
    
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Running predictions...")

    # Run Script 1 (Car Part Predictions)
    car_part_output_path = "Results/predicted_car_part_annos.json"
    car_part_script = 'Model1_largedata/car_part_predictions.py'
    car_part_class_names = 'Model1_largedata/class_names.txt'
    car_part_model_weights = 'Model1_largedata/model_0009999.pth'
    car_part_stdout, car_part_stderr = run_script(car_part_script, image_path, car_part_output_path, car_part_class_names, car_part_model_weights, "Results/car_part_predictions_visualized.jpg")
    
    # Display Script 1 outputs
    with st.expander("Car Part Predictions Output"):
        st.text(car_part_stdout)
        if car_part_stderr:
            st.text("Errors:")
            st.text(car_part_stderr)
    
    # Display Car Part Predictions Image
    if os.path.exists("Results/car_part_predictions_visualized.jpg"):
        car_part_image = Image.open("Results/car_part_predictions_visualized.jpg")
        st.image(car_part_image, caption='Car Part Predictions', use_column_width=True)
    st.success("Car Part Predictions Completed")

    # Run Script 2 (Damage Predictions)
    damage_output_path = "Results/predicted_damage_annos.json"
    damage_script = 'Model2/damage_predictions.py'
    damage_class_names = 'Model2/class_names.txt'
    damage_model_weights = 'Model2/model_0005999.pth'
    damage_stdout, damage_stderr = run_script(damage_script, image_path, damage_output_path, damage_class_names, damage_model_weights, "Results/damage_predictions_visualized.jpg")
    
    # Display Script 2 outputs
    with st.expander("Damage Predictions Output"):
        st.text(damage_stdout)
        if damage_stderr:
            # st.text("Errors:")
            # st.text(damage_stderr)
            print('')
    
    # Display Damage Predictions Image
    if os.path.exists("Results/damage_predictions_visualized.jpg"):
        damage_image = Image.open("Results/damage_predictions_visualized.jpg")
        st.image(damage_image, caption='Damage Predictions', use_column_width=True)
    st.success("Damage Predictions Completed")

    # Run the script to calculate repair costs
    repair_cost_script = 'dice_coefficient_repair_cost.py'
    repair_cost_output_csv = "Results/repair_cost_outputs.csv"
    repair_stdout, repair_stderr = run_script(repair_cost_script, "Results/predicted_damage_annos.json", "Results/predicted_car_part_annos.json", repair_cost_output_csv)
    
    # Display repair cost calculation outputs
    with st.expander("Repair Cost Calculation Output"):
        st.text(repair_stdout)
        if repair_stderr:
            # st.text("Errors:")
            # st.text(repair_stderr)
            print('')
    
    if os.path.exists(repair_cost_output_csv):
        df = pd.read_csv(repair_cost_output_csv)
        if not df.empty and 'total_cost' in df.columns:
            total_cost = df['total_cost'].iloc[0]
            st.subheader(f"Approximate Repair Cost: ${total_cost}")
        else:
            # st.write("No Damage Detected...")
            st.subheader("Please Upload a Alternate/Brighter Image")
    
    # # Read and display total cost
    # if os.path.exists(repair_cost_output_csv):
    #     df = pd.read_csv(repair_cost_output_csv)
    #     total_cost = df['total_cost'].iloc[0]
    #     st.subheader(f"Approximate Repair Cost: ${total_cost}")
    # st.success("Repair Cost Calculation Completed")

    # Save prediction data
    save_prediction_data(uploaded_file.name, car_part_stdout, damage_stdout, repair_stdout)
