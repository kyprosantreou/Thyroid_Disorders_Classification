import os
import cv2
import datetime
import numpy as np
from tqdm import tqdm
from model import UNet
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

# Seeding for reproducibility
seed = 2019
np.random.seed(seed)
tf.random.set_seed(seed)

# Parameters
image_size = 128
base_path = "..\\U-Net_Data"
classes = ["Benign", "Highly_suspicious", "Normal", "Probably_benign", "Suspicious_of_malignancy"]
epochs = 5
batch_size = 8

def get_patient_splits(base_path, classes, train_size=0.85, random_state=23):
    """
    Finds all unique patient IDs across classes and splits them 
    into train and test lists to prevent data leakage.
    """
    all_patients = set()
    
    for cls in classes:
        image_dir = os.path.join(base_path, cls, "images")
        if not os.path.exists(image_dir):
            print(f"Warning: Directory not found -> {image_dir}")
            continue
            
        files = os.listdir(image_dir)
        print(f"Found {len(files)} files in class folder: {cls}")
        
        for image_name in files:
            if image_name.startswith('.'):
                continue
            
            base_name = os.path.splitext(image_name)[0] 
            if "__" in base_name:
                patient_id = base_name.split("__")[0]
            elif "_" in base_name:
                patient_id = base_name.split("_")[0]
            else:
                patient_id = base_name
                
            all_patients.add(f"{cls}_{patient_id}") 
    
    all_patients = list(all_patients)
    print(f"Total unique patients/identifiers found overall: {len(all_patients)}")
    
    if len(all_patients) <= 1:
        raise ValueError(
            f"Could not find enough unique patient entries (Found: {len(all_patients)}). "
            f"Please verify your 'base_path' ({os.path.abspath(base_path)}) and check if your image files exist."
        )
    
    train_patients, test_patients = train_test_split(
        all_patients, train_size=train_size, random_state=random_state
    )
    
    return set(train_patients), set(test_patients)

def load_split_data(base_path, classes, image_size, train_patients, test_patients):
    """
    Loads images and masks directly into their respective 
    train and test sets based on the predefined patient split.
    """
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    
    for cls in classes:
        image_dir = os.path.join(base_path, cls, "images")
        mask_dir = os.path.join(base_path, cls, "masks")
        
        if not os.path.exists(image_dir):
            print(f"Directory not found: {image_dir}")
            continue
            
        print(f"Loading class: {cls}")
        for image_name in tqdm(os.listdir(image_dir)):
            if image_name.startswith('.'):
                continue

            img_path = os.path.join(image_dir, image_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            mask_path = os.path.join(mask_dir, image_name) 
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            img = cv2.resize(img, (image_size, image_size))
            mask = cv2.resize(mask, (image_size, image_size))
            mask = np.expand_dims(mask, axis=-1)
            
            # Fixed: consistent parsing using __ as the primary delimiter
            base_name = os.path.splitext(image_name)[0]  # remove extension first
            if "__" in base_name:
                patient_id = base_name.split("__")[0]
            elif "_" in base_name:
                patient_id = base_name.split("_")[0]
            else:
                patient_id = base_name

            full_patient_key = f"{cls}_{patient_id}"
            
            if full_patient_key in train_patients:
                X_train.append(img)
                Y_train.append(mask)
            elif full_patient_key in test_patients:
                X_test.append(img)
                Y_test.append(mask)
                
    X_train = np.array(X_train) / 255.0
    Y_train = np.array(Y_train) / 255.0
    X_test = np.array(X_test) / 255.0
    Y_test = np.array(Y_test) / 255.0
    
    return X_train, Y_train, X_test, Y_test

# Initialize patient-level data allocation
print("Splitting patients...")
train_patients, test_patients = get_patient_splits(base_path, classes, train_size=0.85, random_state=23)
print(f"Total Patients in Train Set: {len(train_patients)}")
print(f"Total Patients in Test Set: {len(test_patients)}")

# Load data based on patient splits
print("\nLoading images and masks based on patient splits...")
X_train, Y_train, X_test, Y_test = load_split_data(base_path, classes, image_size, train_patients, test_patients)

# Shuffle independently
X_train, Y_train = shuffle(X_train, Y_train, random_state=101)
X_test, Y_test = shuffle(X_test, Y_test, random_state=101)

print("\n--- Final Data Shapes ---")
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# Create and compile model
model = UNet(image_size)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train model
history = model.fit(
    X_train, Y_train, 
    validation_data=(X_test, Y_test), 
    epochs=epochs, 
    batch_size=batch_size, 
    callbacks=[tensorboard_callback]
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f'\nTest Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Save model
model.save("Thyroid_unet_model.keras")
print("Model saved as 'Thyroid_unet_model.keras'")