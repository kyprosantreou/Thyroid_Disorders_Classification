import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

# Parameters
image_size = 128 

# Load the trained model
model = load_model("Thyroid_unet_model.keras")
print("Model loaded successfully.")

def preprocess_image(image_path, image_size=128):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    img = cv2.resize(img, (image_size, image_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_mask(model, image_path, image_size=128):
    processed_image = preprocess_image(image_path, image_size)
    predicted_mask = model.predict(processed_image)
    return np.round(predicted_mask[0]).astype(int)

def compute_gradcam(model, img_array, layer_name):
    grad_model = Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[0]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)[0]
    
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def visualize_prediction(image_path, predicted_mask, heatmap, image_size=128, original_mask_path=None):
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Image not found at {image_path}")

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (image_size, image_size))
    
    # Resize and apply color map to the heatmap to match the input image size
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Create superimposed image
    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap_resized, 0.4, 0)

    # Plot images
    plt.figure(figsize=(20, 5))
    
    # Input image
    plt.subplot(1, 4, 1)
    plt.title("Input Image")
    plt.imshow(original_image)
    plt.axis("off")

    # Original mask
    plt.subplot(1, 4, 2)
    plt.title("Original Mask")
    if original_mask_path is not None:
        original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
        original_mask = cv2.resize(original_mask, (image_size, image_size))
        plt.imshow(original_mask, cmap="gray")
    else:
        plt.text(0.5, 0.5, 'No mask provided', ha='center', va='center')
    plt.axis("off")

    # Predicted mask
    plt.subplot(1, 4, 3)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask.squeeze(), cmap="gray")
    plt.axis("off")

    # Grad-CAM with colorbar
    plt.subplot(1, 4, 4)
    im = plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB), cmap='jet')
    plt.title("Overlayed Image")
    plt.axis("off")

    # Add colorbar
    cbar = plt.colorbar(im, ax=plt.gca())
    cbar.set_label('Heatmap Intensity')

    plt.show()

# Example usage
if __name__ == "__main__":
    test_image_path = "input_image.png"
    original_mask_path = "originalmask.png" 
    
    processed_image = preprocess_image(test_image_path, image_size)
    
    predicted_mask = predict_mask(model, test_image_path, image_size)

    heatmap = compute_gradcam(model, processed_image, layer_name="conv2d_18")

    visualize_prediction(test_image_path, predicted_mask, heatmap, image_size, original_mask_path)