import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import argparse
import sys
import matplotlib.pyplot as plt
import subprocess
from datetime import datetime
import albumentations as A

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# ------------------------- Configuration -------------------------

# Argument parsing
parser = argparse.ArgumentParser(description='Train U-Net model for sunspot detection using PyTorch.')
parser.add_argument('--model-name', type=str, default='default_unet_model', help='Name of the model to be saved.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training.')
parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for optimizer.')
parser.add_argument('--penalty-weight', type=float, default=10.0, help='Penalty weight for Class 1 misclassification.')
parser.add_argument('--img-size',type=int, default=64, help='Tiled image size. Must be divisible by 8.')
args = parser.parse_args()

MODEL_NAME = args.model_name
if MODEL_NAME is None : 
    print("MODEL_NAME argument is empty. Using 'default_unet_model' as argument.")
    sys.stdout.flush()
EPOCHS = args.epochs
if EPOCHS < 1:
    print(f"The number of epochs must be equal or greater than one. You entered {EPOCHS} epochs. ")
    sys.stdout.flush()
    EPOCHS = 1
    print("Number of epochs was corrected to one.")
    sys.stdout.flush()
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
IMG_SIZE = args.img_size
if IMG_SIZE % 8 != 0:
    IMG_SIZE = int(IMG_SIZE - IMG_SIZE%8)
    print(f"Original IMG_SIZE was not divisible by eight. Corrected value : {IMG_SIZE}")
    sys.stdout.flush()
    if IMG_SIZE <=0:
        raise ValueError(f"Correction attempt failed. IMG_SIZE must be divisible by eight and can not be nul or negative. Current value: {IMG_SIZE}")
PENALTY_WEIGHT = args.penalty_weight
if PENALTY_WEIGHT<1:
    raise ValueError(f"PENALTY_WEIGHT must be one or greater. PENALTY_WEIGHT : {PENALTY_WEIGHT}")

# Paths
project_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(project_dir, "dataset")
images_dir = os.path.join(dataset_dir, "images")
annotations_dir = os.path.join(dataset_dir, "annotations")
annotations_path = os.path.join(annotations_dir, "instances_default.json")
masks_dir = os.path.join(dataset_dir, "masks")
output_dir = os.path.join(project_dir, "output")
trained_models_dir = os.path.join(output_dir, "trained_models")
os.makedirs(trained_models_dir, exist_ok=True)
logs_dir = os.path.join(output_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Redirect stdout to log file
log_path = os.path.join(logs_dir, 'training.log')
sys.stdout = open(log_path, 'w', encoding='utf-8')

print(f"Model name : {MODEL_NAME}")
sys.stdout.flush()
print("")
print("------------------------- Initialization : -------------------------")
print(f"Number of epochs : {EPOCHS}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Learning rate : {LEARNING_RATE}")
print(f"Tilling image size : {IMG_SIZE}")
print(f"Penalty weight for class 1 : {PENALTY_WEIGHT}")
print("")
sys.stdout.flush()

print("[INFO] Training started at :",datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
sys.stdout.flush()
print("Logging configuration completed.")
sys.stdout.flush()

# ------------------------- GPU Setup -------------------------
print("")
print("------------------------- GPU Setup -------------------------")
sys.stdout.flush()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
sys.stdout.flush()

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

# ------------------------- Error Handling and Setup -------------------------
def check_and_create_paths():
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"[INFO] Created dataset directory at '{dataset_dir}'. Please add your images and annotations.")
        sys.stdout.flush()
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"[INFO] Created images directory at '{images_dir}'. Please add your images.")
        sys.stdout.flush()
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)
        print(f"[INFO] Created annotations directory at '{annotations_dir}'. Please add your annotations.")
        sys.stdout.flush()
    
    if len(os.listdir(images_dir)) == 0:
        print(f"[ERROR] No images found in '{images_dir}'. Please ensure the directory is not empty.")
        sys.stdout.flush()
        sys.exit(1)
    if not os.path.exists(annotations_path):
        print(f"[ERROR] Annotations file not found at '{annotations_path}'. Please provide the annotations.")
        sys.stdout.flush()
        sys.exit(1)
    
    if not os.path.exists(masks_dir) or len(os.listdir(masks_dir)) == 0:
        print("[INFO] Masks not found. Generating masks from annotations...")
        sys.stdout.flush()
        os.makedirs(masks_dir, exist_ok=True)
        generate_masks()

def generate_masks():
    convert_script = os.path.join(project_dir, 'convert_annotations.py')
    if not os.path.exists(convert_script):
        print(f"[ERROR] convert_annotations.py script not found at '{convert_script}'.")
        sys.stdout.flush()
        sys.exit(1)
    command = f'python "{convert_script}" --images-dir "{images_dir}" --annotations-path "{annotations_path}" --masks-dir "{masks_dir}"'
    subprocess.run(command, shell=True)
    print("[INFO] Masks generated successfully.")
    sys.stdout.flush()



# Augmentation 1: Rotate + Salt and Pepper (using Lambda)
aug_transform_1 = A.Compose([
    A.Rotate(limit=30, p=1.0),
    A.SaltAndPepper(amount=(0.01,0.04),salt_vs_pepper=(0.1,0.2),p=1)
])

# Augmentation 2: Rotate + Blur (low)
aug_transform_2 = A.Compose([
    A.Rotate(limit=30, p=1.0),
    A.Blur(blur_limit=2, p=1.0)
])

# Augmentation 3: Rotate + Under Exposure (slight)
aug_transform_3 = A.Compose([
    A.Rotate(limit=30, p=1.0),
    A.RandomBrightnessContrast (  brightness_limit=  (-0.4,0.1), p=1.0)
])



class SunspotDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        super().__init__()
        self.images = []
        self.masks = []
        
        image_paths = []
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    image_paths.append(os.path.join(root, file))
        
        mask_paths = []
        for root, _, files in os.walk(masks_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    mask_paths.append(os.path.join(root, file))
        
        image_name_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
        mask_name_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}
        
        self.common_files = list(set(image_name_to_path.keys()) & set(mask_name_to_path.keys()))
        
        if not self.common_files:
            print("[ERROR] No matching images and masks found. Ensure filenames match.")
            sys.stdout.flush()
            sys.exit(1)
        
        self.image_name_to_path = image_name_to_path
        self.mask_name_to_path = mask_name_to_path
        
        # Each original image will yield 4 samples: original + 3 augmented
        self.total_base_images = len(self.common_files)
        self.total_images = self.total_base_images * 4

    def __len__(self):
        return self.total_images
    
    def __getitem__(self, idx):
        base_idx = idx // 4
        variant = idx % 4

        filename = self.common_files[base_idx]
        image_path = self.image_name_to_path[filename]
        mask_path = self.mask_name_to_path[filename]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = image.resize((IMG_SIZE, IMG_SIZE))
        mask = mask.resize((IMG_SIZE, IMG_SIZE))

        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0

        # Apply augmentation depending on variant
        if variant == 0:
            # Original
            # no augmentation
            pass
        elif variant == 1:
            # Rotate + Salt & Pepper
            augmented = aug_transform_1(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        elif variant == 2:
            # Rotate + Blur
            augmented = aug_transform_2(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        elif variant == 3:
            # Rotate + Under exposure
            augmented = aug_transform_3(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = np.expand_dims(mask, axis=0)
        image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        return image, mask, filename

# ------------------------- U-Net Model -------------------------
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
        return outputs

# ------------------------- Loss and Optimizer -------------------------
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # convert logits to probabilities
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2)

    dice = (2. * intersection + smooth) / (union + smooth)
    loss = 1 - dice
    return loss.mean()


def custom_loss(pred, target):
    global PENALTY_WEIGHT
    # BCEWithLogitsLoss expects raw logits. pred here is logits because we removed sigmoid from forward
    class_weights = torch.tensor([1.0, PENALTY_WEIGHT], device=device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])(pred, target)
    d_loss = dice_loss(pred, target)  # dice_loss applies sigmoid internally now
    total_loss = d_loss + bce_loss
    return total_loss


# ------------------------- Training -------------------------
print("")
print("------------------------- Training -------------------------")
sys.stdout.flush()

def train_model(model, train_loader, val_loader, criterion, optimizer):
    history = {'loss': [], 'val_loss': []}
    total_steps = len(train_loader)
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (images, masks, filenames) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}")
                sys.stdout.flush()
                print(f"Processing images: {filenames}")
                sys.stdout.flush()
        
        avg_train_loss = running_loss / total_steps
        history['loss'].append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        sys.stdout.flush()
    print("")
    print("------------------------- Save Model and Plot -------------------------")
    sys.stdout.flush()
    print("[INFO] Training completed successfully.")
    sys.stdout.flush()
    return history

def save_model_and_plot(model, history):
    model_save_path = os.path.join(trained_models_dir, f"{MODEL_NAME}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"[INFO] Model saved to {model_save_path}")
    sys.stdout.flush()
    
    plt.figure()
    epochs_range = range(1, EPOCHS + 1)
    loss = history['loss']
    val_loss = history['val_loss']
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.suptitle(f"Training and Validation Loss for {MODEL_NAME}", fontweight="bold")
    plt.title(f"Trained on {len(train_dataset)} images, evaluated on {len(val_dataset)} images.", fontsize=10)
    plt.legend()
    plot_save_path = os.path.join(output_dir, f"{MODEL_NAME}_learning_curve.png")
    plt.savefig(plot_save_path)
    print(f"[INFO] Training plot saved to {plot_save_path}")
    sys.stdout.flush()
    
    history_save_path = os.path.join(output_dir, f"{MODEL_NAME}_history.json")
    with open(history_save_path, 'w') as f:
        json.dump(history, f)
    print(f"[INFO] Training history saved to {history_save_path}")
    sys.stdout.flush()

if __name__ == "__main__":
    print("[INFO] Checking and setting up paths.")
    sys.stdout.flush()
    check_and_create_paths()
    
    print("[INFO] Loading dataset...")
    sys.stdout.flush()
    dataset = SunspotDataset(images_dir, masks_dir)
    
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"[INFO] Starting training on {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    print("")
    sys.stdout.flush()
    
    model = UNet().to(device)
    criterion = custom_loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = train_model(model, train_loader, val_loader, criterion, optimizer)
    save_model_and_plot(model, history)
    
    print("[INFO] Training process completed successfully.")
    sys.stdout.flush()
    print("")
    print("[INFO] Training ended at :",datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    sys.stdout.flush()
