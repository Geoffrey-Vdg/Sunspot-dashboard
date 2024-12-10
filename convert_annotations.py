# convert_annotations.py

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import argparse

def create_masks_from_cvat_annotations(images_dir, annotations_path, masks_dir):
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Create a mapping from image ID to file name
    image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}
    
    # Ensure masks directory exists
    os.makedirs(masks_dir, exist_ok=True)
    
    # Initialize a dictionary to store all annotations for each image
    image_annotations = {}
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Process each image
    for image_id, anns in image_annotations.items():
        filename = image_id_to_filename[image_id]
        image_path = os.path.join(images_dir, filename)
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"[WARNING] Image file '{image_path}' not found. Skipping.")
            continue
        
        # Open the image to get its dimensions
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Create a blank mask
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw each polygon
        for ann in anns:
            if ann.get('iscrowd', 0):
                # Handle crowd annotations if necessary
                continue
            segmentation = ann['segmentation']
            if isinstance(segmentation, list):
                # Polygon format
                for polygon in segmentation:
                    xy = np.array(polygon).reshape((-1, 2))
                    draw.polygon(xy.flatten().tolist(), outline=255, fill=255)
            elif isinstance(segmentation, dict) and segmentation.get('counts'):
                # RLE format - not handled here
                continue
        
        # Save the mask
        # Adjust the mask filename to match the image filename (without directories)
        mask_filename = os.path.splitext(os.path.basename(filename))[0] + '.png'
        mask_save_path = os.path.join(masks_dir, mask_filename)
        mask.save(mask_save_path)
        print(f"[INFO] Saved mask for image '{filename}' to '{mask_save_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CVAT polygon annotations to mask images.')
    parser.add_argument('--images-dir', type=str, required=True, help='Directory containing the images.')
    parser.add_argument('--annotations-path', type=str, required=True, help='Path to the annotations JSON file.')
    parser.add_argument('--masks-dir', type=str, required=True, help='Directory to save the generated masks.')
    args = parser.parse_args()
    
    create_masks_from_cvat_annotations(args.images_dir, args.annotations_path, args.masks_dir)
