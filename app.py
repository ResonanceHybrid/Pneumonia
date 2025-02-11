from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import cv2
from tensorflow.keras.models import load_model
import json
from model.utils import (
    load_and_preprocess_image, generate_gradcam, 
    create_heatmap_overlay, load_model_metrics,
    calculate_ensemble_prediction
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# Load models
custom_cnn = load_model('models/custom_cnn_final.keras')
resnet101 = load_model('models/resnet101_final.keras')

@app.route('/')
def home():
    metrics = load_model_metrics()
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Save and process image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Prepare images for both models
        img_gray = load_and_preprocess_image(filepath, is_rgb=False)
        img_rgb = load_and_preprocess_image(filepath, is_rgb=True)
        
        # Get predictions
        custom_pred = float(custom_cnn.predict(img_gray)[0][0])
        resnet_pred = float(resnet101.predict(img_rgb)[0][0])
        
        # Generate GradCAM visualizations
        custom_heatmap = generate_gradcam(custom_cnn, img_gray, 'conv2d_7')  
        resnet_heatmap = generate_gradcam(resnet101, img_rgb, 'resnet101v2') 
        
        # Create and save heatmap overlays
        original_img = cv2.imread(filepath)
        original_img = cv2.resize(original_img, (224, 224))
        
        custom_overlay = create_heatmap_overlay(original_img, custom_heatmap)
        resnet_overlay = create_heatmap_overlay(original_img, resnet_heatmap)
        
        custom_heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], f'custom_heatmap_{filename}')
        resnet_heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], f'resnet_heatmap_{filename}')
        
        cv2.imwrite(custom_heatmap_path, custom_overlay)
        cv2.imwrite(resnet_heatmap_path, resnet_overlay)
        
        # Calculate ensemble prediction
        ensemble_result = calculate_ensemble_prediction(custom_pred, resnet_pred)
        
        # Continuing from previous app.py...
        return jsonify({
            'custom_cnn': {
                'probability': custom_pred,
                'prediction': 'Pneumonia' if custom_pred > 0.5 else 'Normal',
                'heatmap': url_for('static', filename=f'uploads/custom_heatmap_{filename}')
            },
            'resnet101': {
                'probability': resnet_pred,
                'prediction': 'Pneumonia' if resnet_pred > 0.5 else 'Normal',
                'heatmap': url_for('static', filename=f'uploads/resnet_heatmap_{filename}')
            },
            'ensemble': ensemble_result,
            'original_image': url_for('static', filename=f'uploads/{filename}')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)