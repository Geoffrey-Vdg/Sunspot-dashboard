import os
import subprocess
import sys

# Complete list of required dependencies
DEPENDENCIES = [
    "dash",
    "dash-bootstrap-components",
    "Pillow",  # PIL library including Image and ImageDraw
    "numpy",
    "torch",
    "torchvision",
    "matplotlib",
    "albumentations",
    "opencv-python-headless",
    "argparse",
    "json5",  # For JSON handling
    "pybase64",  # For base64 operations
]

# Directories and files to check/create
REQUIRED_DIRECTORIES = [
    "dataset",
    "dataset/images",
    "dataset/annotations",
    "dataset/masks",
    "output",
    "output/trained_models",
    "output/logs",
    "output/segmented_dataset",
    "images_to_run",
]
REQUIRED_FILES = [
    "convert_annotations.py",  # Placeholder for script mentioned in train_cnn.py
]

def install_dependencies():
    print("[INFO] Installing required dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + DEPENDENCIES)
    print("[INFO] Dependencies installed successfully.")

def check_and_create_directories():
    print("[INFO] Checking and creating required directories...")
    for directory in REQUIRED_DIRECTORIES:
        os.makedirs(directory, exist_ok=True)
        print(f"[INFO] Directory checked/created: {directory}")
    print("[INFO] All required directories are ready.")

def check_and_create_files():
    print("[INFO] Checking and creating required files...")
    for file in REQUIRED_FILES:
        if not os.path.exists(file):
            with open(file, "w") as f:
                f.write("# Placeholder script for annotation conversion.")
            print(f"[INFO] File created: {file}")
        else:
            print(f"[INFO] File exists: {file}")
    print("[INFO] All required files are ready.")

def main():
    try:
        install_dependencies()
        check_and_create_directories()
        check_and_create_files()
        print("[INFO] Setup completed successfully.")
    except Exception as e:
        print(f"[ERROR] Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
