# Sunspot Detection Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![COCO Format](https://img.shields.io/badge/Dataset-COCO-green)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red)



**Author:** Vandegaer Geoffrey


## Overview

The **Sunspot Detection Dashboard** provides a framework for managing labeled datasets in COCO format to train a customized U-Net model for sunspot detection. It supports the following functionalities:

1. Upload labeled images for training a tailored U-Net model.
2. Upload images to perform inferences using a pre-trained model.
3. Visualize segmentation results and track training progress interactively.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [License](#license)

---

## Installation

Before the first use, run the `install.py` script. This script ensures that all necessary packages are installed, verifies the file structure, and confirms the presence of required scripts.

---

## File Structure

The intended project directory is structured as follows:

```bash
project_dir/                          # Root directory
├── train_cnn.py                      # Script for training the model
├── convert_annotations.py            # Converts annotations to masks
├── run_inference.py                  # Performs inference using the trained model
├── dashboard.py                      # Launches the interactive dashboard
├── install.py                        # Installation script
├── detect_labeled_images.py          # (Optional) Filters labeled images
├── dataset/                          # Dataset directory
│   ├── images/                       # Input images
│   ├── annotations/                  # COCO JSON annotation file
│   └── masks/                        # Pixel masks generated from annotations
├── output/                           # Output directory
│   ├── trained_models/               # Saved models
│   ├── logs/                         # Logs for training
│   ├── training_plots/               # Training loss and accuracy plots
│   └── inference_results/            # Results from inference
├── images_to_run/                    # Images for inference
└── segmented_dataset/                # Segmentation results
```


Key Notes:

- The **dataset** folder contains `images`, `annotations`, and `masks`. 
  - Annotations must be in COCO format, named `instances_default.json`.
  - Masks are generated from polygon labels in the JSON file using `convert_annotations.py`.
- The **output** folder stores trained models, logs, training plots, and inference results.

---

## Usage

### Dashboard Overview

The dashboard consists of three pages:

1. **Dataset Page**
   - Upload labeled images and COCO annotation files for training.
   - Upload images for inference using a trained model.

2. **Train Page**
   - Configure training parameters, including:
     - Model name
     - Number of epochs
     - Batch size
     - Learning rate
     - Tiled image size
     - Class 1 penalty weight (false negatives for sunspots)
   - View real-time training progress and logs:
     - Selected parameters
     - Training/validation loss per epoch
     - Training completion status
   - Visualize training and validation loss plots.

3. **Inference Page**
   - Select a trained model for inference.
   - Run inferences on uploaded images.
   - Compare original images with segmentation results and masks.


### Steps to Use

1. **Upload Dataset**: 
   - On the **Dataset Page**, upload labeled images and a COCO annotation file (`instances_default.json`). Ensure all files adhere to the specified format.
   - To perform inference, upload images in the "Dataset for Inference" section.

2. **Optional Filtering**:
   - If your dataset contains unlabeled images, use `detect_labeled_images.py` to filter them. Update the `dataset/images` and `dataset/annotations` folders accordingly.

3. **Train the Model**:
   - Navigate to the **Train Page**.
   - Configure training parameters and start training.

4. **Perform Inference**:
   - After training is complete, go to the **Inference Page**.
   - Select the trained model, run inference, and explore the segmentation results visually.

### Architecture 

![Dashboard Overview](assets/Sunspot-dashboard(demo).png)

### Image Pipeline 

**Train_cnn.py image pipeline :**

<div style="display: flex; justify-content: space-around;">
  <div>
    <p>Step 1: Original Image</p>
    <img src="assets/usd195604170830.bmp" width="200">
  </div>
  <div>
    <p>Step 2: Mask Original Image</p>
    <img src="assets/usd195604170830.png" width="200">
  </div>
  <div>
    <p>Step 3: Augmented Images</p>
    <img src="assets/Augmented_images.png" width="200">
  </div>
</div>

<br>
<br>

**run_inference.py image pipeline :**

<div style="display: flex; justify-content: space-around;">
  <div>
    <p>Step 1: Original Image</p>
    <img src="assets/usd198202101052.jpg" width="200">
  </div>
  <div>
    <p>Step 2: Square Crop Original Image</p>
    <img src="assets/usd198202101052_square.jpg" width="200">
  </div>
  <div>
    <p>Step 3: Darken Out of Circle Pixels</p>
    <img src="assets/usd198202101052_square_darken.png" width="200">
  </div>
</div>

<div style="display: flex; justify-content: space-around;">
  <div>
    <p>Step 4: Mask of detected sunspots</p>
    <img src="assets/usd198202101052_square_mask.png" width="200">
  </div>
  <div>
    <p>Step 5: Expand the mask </p>
    <img src="assets/Full_mask.png" width="200">
  </div>
</div>


---

## License

**All rights reserved.**

---

## Note

This dashboard is an initial version developed as a proof of concept. Future updates will focus on refining the architecture and improving functionality based on user feedback and further testing.

