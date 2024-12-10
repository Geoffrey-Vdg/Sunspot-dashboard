import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import argparse
import sys
import csv
import warnings
from datetime import datetime
import json

# Suppress FutureWarnings related to torch.load
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


# ------------------------- Configuration -------------------------

parser = argparse.ArgumentParser(description='Run inference using the trained U-Net model.')
parser.add_argument('--model-name',default='default_unet_model', type=str, help='Name of the model to use for inference.')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for converting probabilities to binary mask.')
args = parser.parse_args()

MODEL_NAME = args.model_name
THRESHOLD = args.threshold
IMG_SIZE = 64  # Should match the size used during training: MUST be divisible by 8
if IMG_SIZE % (8) != 0:
    IMG_SIZE = IMG_SIZE-IMG_SIZE%8
    print(f"Original IMG_SIZE was not divisible by eight. Corrected value : {IMG_SIZE}")
    sys.stdout.flush()
    if IMG_SIZE <=0:
        raise ValueError(f"Correction attempt failed. IMG_SIZE must be divisible by eight and can not be nul or negative. Current value: {IMG_SIZE}")
        sys.stdout.flush()

# Paths
project_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(project_dir, "images_to_run")
output_dir = os.path.join(project_dir, "output")
segmented_dir = os.path.join(output_dir, "segmented_dataset")
trained_models_dir = os.path.join(output_dir, "trained_models")
os.makedirs(segmented_dir, exist_ok=True)

# Update the model path to match the specified location
model_path = os.path.join(trained_models_dir, f"{MODEL_NAME}.pth")

csv_path = os.path.join(output_dir, "sunspot_table.csv")

# Redirect stdout to log file
log_path = os.path.join(output_dir, 'inference.log')
sys.stdout = open(log_path, 'w', encoding='utf-8')
print(f"Model used : {MODEL_NAME}")
print("[INFO] Inference started at :",datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
sys.stdout.flush()

# ------------------------- U-Net Model Definition -------------------------

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        def CBR(in_channels, out_channels):
            layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            return layers
        
        self.enc1 = CBR(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = CBR(256, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)
        
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        p1 = self.pool1(enc1)
        enc2 = self.enc2(p1)
        p2 = self.pool2(enc2)
        enc3 = self.enc3(p2)
        p3 = self.pool3(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(p3)
        
        # Decoder
        up3 = self.upconv3(bottleneck)
        cat3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(cat3)
        up2 = self.upconv2(dec3)
        cat2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(cat2)
        up1 = self.upconv1(dec2)
        cat1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(cat1)
        
        outputs = self.conv_last(dec1)
        outputs = torch.sigmoid(outputs)
        return outputs

# ------------------------- Load Model -------------------------
print("")
print("------------------------- Load Model -------------------------")
sys.stdout.flush()

print("[INFO] Loading model...")
sys.stdout.flush()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {device}")
sys.stdout.flush()
print("")

if device.type == 'cuda':
    print("CUDA is available!")
    sys.stdout.flush()
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")
    for device_id in range(num_devices):
        print(f"\nDevice ID: {device_id}")
        print(f"Device Name: {torch.cuda.get_device_name(device_id)}")
        print(f"Device Capability: {torch.cuda.get_device_capability(device_id)}")
        print(f"Device Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")
        sys.stdout.flush()
else:
    print("CUDA is not available. Training will use the CPU.")
    sys.stdout.flush()


model = UNet().to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[INFO] Model '{MODEL_NAME}' loaded successfully.")
    sys.stdout.flush()
except Exception as e:
    print(f"[ERROR] Failed to load model '{MODEL_NAME}': {e}")
    sys.stdout.flush()
    sys.exit(1)



# ------------------------- Cropping Function -------------------------


def detect_circle_and_crop(image_path):
    """
    Detect the largest circle in the image, crop around it, and blacken pixels outside the circle.

    Returns:
    - cropped_image: The cropped image with outside pixels blackened.
    - circle_params: A dictionary with the center and radius of the detected circle.
    - crop_offsets: The top-left corner of the cropping box in the original image.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=200
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0], key=lambda c: c[2])  # Max radius
        x, y, radius = largest_circle

        # Define cropping region
        n = int(radius * 2)
        half_n = n // 2
        start_x = max(x - half_n, 0)
        start_y = max(y - half_n, 0)
        end_x = min(x + half_n, image.shape[1])
        end_y = min(y + half_n, image.shape[0])
        crop_offsets = (start_x, start_y)

        # Crop the image
        cropped_image = image[start_y:end_y, start_x:end_x]

        # Blacken pixels outside the circle
        mask = np.zeros_like(cropped_image)
        cv2.circle(mask, (radius, radius), radius, (255, 255, 255), -1)
        cropped_image = cv2.bitwise_and(cropped_image, mask)

        return cropped_image, {"center": (x, y), "radius": radius}, crop_offsets

    return None, None, None

# ------------------------- Mapping Function -------------------------
def map_coordinates_to_original(cropped_coords, crop_offsets):
    """
    Map coordinates from the cropped image back to the original image.

    Parameters:
    - cropped_coords: List of tuples [(x1, y1), ...] in cropped image.
    - crop_offsets: Top-left corner (x_offset, y_offset) of the cropping box.

    Returns:
    - original_coords: List of tuples [(x1, y1), ...] in the original image.
    """
    x_offset, y_offset = crop_offsets
    original_coords = [(x + x_offset, y + y_offset) for x, y in cropped_coords]
    return original_coords


# ------------------------- Compute Feature Coordinates -------------------------

def compute_feature_coordinates(label_image, circle_center):
    """
    Compute the polar and Cartesian coordinates of each feature in the label image.

    Parameters:
    - label_image: 2D numpy array where each feature has a unique label ID.
    - circle_center: Tuple (center_x, center_y) of the circle.

    Returns:
    - feature_list: List of dictionaries containing labelID, Cartesian, and polar coordinates.
    """
    features = []
    center_x, center_y = circle_center
    unique_labels = np.unique(label_image)
    
    # Skip the background label (0)
    for labelID in unique_labels:
        if labelID == 0:
            continue

        # Find all pixels belonging to the current label
        y_coords, x_coords = np.where(label_image == labelID)
        if len(x_coords) == 0 or len(y_coords) == 0:
            continue

        # Compute the centroid of the feature
        x_feature = np.mean(x_coords)
        y_feature = np.mean(y_coords)
        
        # Compute polar coordinates relative to the circle center
        r = np.sqrt((x_feature - center_x)**2 + (y_feature - center_y)**2)
        theta = np.arctan2(y_feature - center_y, x_feature - center_x)  # In radians

        # Append feature data
        features.append({
            "labelID": int(labelID),
            "polar_coordinate_system": {
                "radius": r,
                "angle": theta  # Keep in radians for precision
            },
            "cartesian_coordinate_system": {
                "x": float(x_feature),
                "y": float(y_feature)
            }
        })

    return features


# ------------------------- Generate JSON -------------------------

def generate_json(image_paths, circle_params_dict, segmented_dir, output_dir):
    """
    Generate a JSON file containing feature data for each image.

    Parameters:
    - image_paths: List of image file paths.
    - circle_params_dict: Dictionary mapping image paths to circle parameters.
    - segmented_dir: Directory containing segmented masks.
    - output_dir: Directory to save the JSON file.

    Returns:
    - json_path: Path to the saved JSON file.
    """
    results = {}
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        print(f"[INFO] Processing JSON data for '{filename}'...")
        
        # Read the mask and detect features
        mask_path = os.path.join(segmented_dir, os.path.splitext(filename)[0] + "_mask.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        num_labels, label_image = cv2.connectedComponents(mask)
        
        # Get the circle center for the current image
        circle_center = circle_params_dict[filename]["center"]
        
        # Compute feature coordinates
        feature_list = compute_feature_coordinates(label_image, circle_center)
        
        # Add to results
        results[filename] = feature_list

    # Save JSON
    json_path = os.path.join(output_dir, "features.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[INFO] JSON saved to '{json_path}'")
    return json_path

# ------------------------- Inference Function -------------------------

def process_image_with_cropping(image_path):
    """
    Process an image: detect circle, crop, blacken, run inference, and map back coordinates.

    Returns:
    - original_image: The original image with labeled features overlaid.
    - final_mask: The full-size mask in the original image dimensions.
    """
    # Detect and crop the circle
    cropped_image, circle_params, crop_offsets = detect_circle_and_crop(image_path)
    if cropped_image is None:
        print(f"No circle detected in '{image_path}'. Skipping...")
        return None, None

    # Resize cropped image to model input size
    resized_cropped = cv2.resize(cropped_image, (IMG_SIZE, IMG_SIZE))
    resized_cropped = resized_cropped / 255.0  # Normalize
    input_tensor = torch.tensor(resized_cropped.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        prediction = model(input_tensor)[0].cpu().numpy()[0]  # Prediction mask

    # Resize prediction back to cropped size
    prediction_resized = cv2.resize(prediction, (cropped_image.shape[1], cropped_image.shape[0]))

    # Create a full-size mask initialized to zero
    original_image = cv2.imread(image_path)
    final_mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)

    # Map prediction back to the original image
    for (y_cropped, x_cropped), value in np.ndenumerate(prediction_resized):
        # Compute original image coordinates
        x_original = x_cropped + crop_offsets[0]
        y_original = y_cropped + crop_offsets[1]

        # Set the mask value
        if 0 <= x_original < final_mask.shape[1] and 0 <= y_original < final_mask.shape[0]:
            final_mask[y_original, x_original] = 255 if value > THRESHOLD else 0

    # Overlay the detected features on the original image
    labeled_image = original_image.copy()
    for (y, x) in zip(*np.where(final_mask > 0)):
        cv2.circle(labeled_image, (x, y), 2, (0, 0, 255), -1)  # Mark detected features

    return labeled_image, final_mask

# ------------------------- Main Inference Loop -------------------------
print("")
print("------------------------- Inference Loop -------------------------")
sys.stdout.flush()
# ------------------------- Main Inference Loop -------------------------
print("")
print("------------------------- Inference Loop -------------------------")
sys.stdout.flush()
def main():
    # Gather all image file paths
    image_files = []
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"[ERROR] No images found in '{images_dir}'. Please add images for inference.")
        sys.stdout.flush()
        sys.exit(1)
    
    print(f"[INFO] Starting inference on {len(image_files)} images...")
    sys.stdout.flush()
    
    csv_rows = []
    feature_data = {}  # Dictionary to store JSON data
    for image_path in image_files:
        filename = os.path.basename(image_path)
        print(f"[INFO] Processing image '{filename}'...")
        sys.stdout.flush()
    
        # Process image: get labeled image and final mask
        labeled_image, final_mask = process_image_with_cropping(image_path)
        if labeled_image is None or final_mask is None:
            print(f"[WARNING] Skipping image '{filename}' due to processing issues.")
            sys.stdout.flush()
            continue

        # Save mask
        mask_filename = os.path.splitext(filename)[0] + '_mask.png'
        mask_save_path = os.path.join(segmented_dir, mask_filename)
        cv2.imwrite(mask_save_path, final_mask)
        print(f"[INFO] Saved mask to '{mask_save_path}' (overwriting if exists).")
        sys.stdout.flush()

        # Save labeled original image
        overlay_filename = os.path.splitext(filename)[0] + '_overlay.png'
        overlay_save_path = os.path.join(segmented_dir, overlay_filename)
        cv2.imwrite(overlay_save_path, labeled_image)
        print(f"[INFO] Saved labeled image to '{overlay_save_path}' (overwriting if exists).")
        sys.stdout.flush()
    
        # Count the number of sunspots in the mask
        num_labels, labels_im = cv2.connectedComponents(final_mask)
        num_sunspots = num_labels - 1  # Subtract 1 for the background label

        # Compute feature data for JSON
        cropped_image, circle_params, crop_offsets = detect_circle_and_crop(image_path)
        if circle_params is not None:
            circle_center = circle_params["center"]
            features = compute_feature_coordinates(labels_im, circle_center)
            feature_data[filename] = features  # Add feature data for JSON

        # Extract date and time from filename (if applicable)
        name = filename
        date = "Unknown"
        time = "Unknown"

        # Add row to CSV data
        csv_rows.append({
            "name": name,
            "date": date,
            "time": time,
            "sunspot": num_sunspots
        })
    
        print(f"[INFO] Detected {num_sunspots} sunspots in image '{filename}'.")
        sys.stdout.flush()
    
    print("")
    print("------------------------- Saving inferences -------------------------")
    sys.stdout.flush()
    print("---- CSV file counting features ----")
    sys.stdout.flush()
    # Write CSV file
    csv_headers = ["name", "date", "time", "sunspot"]
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    
    print(f"[INFO] CSV file saved to '{csv_path}'.")
    print("[INFO] Inference completed.")
    sys.stdout.flush()
    print("")
    print("---- Json file with polar coordinates ----")
    sys.stdout.flush()
    # Save JSON file
    json_path = os.path.join(output_dir, f"{MODEL_NAME}_features_polar_coordinates.json")
    with open(json_path, "w") as f:
        json.dump(feature_data, f, indent=4)
    print(f"[INFO] JSON saved to '{json_path}'.")
    sys.stdout.flush()
    print("")

    print("Inference ended at :", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
