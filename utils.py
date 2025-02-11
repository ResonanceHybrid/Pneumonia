import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import tensorflow as tf
import json
import os

def load_and_preprocess_image(image_path, target_size=(224, 224), is_rgb=False):
    """Load and preprocess a single image"""
    if is_rgb:
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    
    if not is_rgb:
        img = np.expand_dims(img, axis=-1)
    return np.expand_dims(img, axis=0)

def generate_gradcam(model, img_array, last_conv_layer):
    """Generate Grad-CAM visualization"""
    try:
        # Find appropriate convolutional layer
        if last_conv_layer == 'conv2d_7':
            # For custom CNN, find the last convolutional layer
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer.name
                    break
        elif last_conv_layer == 'resnet101v2':
            # For ResNet, find the last convolutional layer
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer.name
                    break
        
        # Create grad model
        try:
            grad_model = tf.keras.models.Model(
                [model.inputs],
                [model.get_layer(last_conv_layer).output, model.output]
            )
        except ValueError as e:
            print(f"Layer not found: {last_conv_layer}. Using last conv layer instead.")
            # Fallback to last convolutional layer
            conv_layers = [layer.name for layer in model.layers 
                         if isinstance(layer, tf.keras.layers.Conv2D)]
            if not conv_layers:
                raise ValueError("No convolutional layers found in the model")
            last_conv_layer = conv_layers[-1]
            grad_model = tf.keras.models.Model(
                [model.inputs],
                [model.get_layer(last_conv_layer).output, model.output]
            )
        
        # Generate gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        
        # Calculate gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Generate heatmap
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    except Exception as e:
        print(f"Error generating Grad-CAM: {str(e)}")
        # Return a blank heatmap in case of error
        return np.zeros((7, 7))  # Default size for visualization

def create_heatmap_overlay(original_image, heatmap, alpha=0.4):
    """Create heatmap overlay on original image"""
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    superimposed_img = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
    return superimposed_img

def load_model_metrics():
    """Load model metrics from saved files"""
    metrics = {}
    default_metrics = {
        'test_metrics': {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    }
    
    for model_name in ['custom_cnn', 'resnet101']:
        metrics_file = f'static/images/{model_name}_metrics.json'
        
        # If the file exists, load it and ensure correct structure
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    model_data = json.load(f)
                    # Check if the loaded data has the required structure
                    if 'test_metrics' not in model_data:
                        model_data = {'test_metrics': model_data}
                    # Ensure all required metrics exist
                    if not all(key in model_data['test_metrics'] for key in ['accuracy', 'precision', 'recall', 'f1_score']):
                        model_data = default_metrics
                    metrics[model_name] = model_data
            except (json.JSONDecodeError, FileNotFoundError):
                metrics[model_name] = default_metrics
        else:
            metrics[model_name] = default_metrics
            
    return metrics

def calculate_ensemble_prediction(custom_pred, resnet_pred):
    """Calculate ensemble prediction"""
    ensemble_prob = (custom_pred + resnet_pred) / 2
    return {
        'probability': float(ensemble_prob),
        'prediction': 'Pneumonia' if ensemble_prob > 0.5 else 'Normal'
    }