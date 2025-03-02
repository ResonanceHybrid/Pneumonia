import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout, 
    BatchNormalization, Input, GlobalAveragePooling2D
)
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import json
import shutil
from pathlib import Path

def prepare_data_splits(data_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    """Prepare stratified train, validation, and test splits only if they don't exist"""
    assert train_ratio + valid_ratio + test_ratio == 1.0
    
    base_path = Path(data_dir).parent
    splits = {
        'train': base_path / 'train_split',
        'valid': base_path / 'valid_split',
        'test': base_path / 'test_split'
    }
    
    # Check if all split directories exist and contain data
    splits_exist = all(
        split_dir.exists() and 
        (split_dir / 'NORMAL').exists() and 
        (split_dir / 'PNEUMONIA').exists() and
        len(list((split_dir / 'NORMAL').glob('*.jpeg'))) > 0 and
        len(list((split_dir / 'PNEUMONIA').glob('*.jpeg'))) > 0
        for split_dir in splits.values()
    )
    
    if splits_exist:
        print("Data splits already exist. Using existing splits...")
        return str(splits['train']), str(splits['valid']), str(splits['test'])
    
    print("Creating new data splits...")
    # Create directories
    for split_dir in splits.values():
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for class_name in ['NORMAL', 'PNEUMONIA']:
            (split_dir / class_name).mkdir(parents=True)
    
    # Split data with stratification
    for class_name in ['NORMAL', 'PNEUMONIA']:
        files = list((Path(data_dir) / class_name).glob('*.jpeg'))
        np.random.shuffle(files)
        
        n_files = len(files)
        n_train = int(n_files * train_ratio)
        n_valid = int(n_files * valid_ratio)
        
        train_files = files[:n_train]
        valid_files = files[n_train:n_train + n_valid]
        test_files = files[n_train + n_valid:]
        
        # Copy files
        for f in train_files:
            shutil.copy2(f, splits['train'] / class_name / f.name)
        for f in valid_files:
            shutil.copy2(f, splits['valid'] / class_name / f.name)
        for f in test_files:
            shutil.copy2(f, splits['test'] / class_name / f.name)
        
        print(f"Split {class_name} images - Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")
    
    return str(splits['train']), str(splits['valid']), str(splits['test'])

def create_custom_cnn():
    """Improved Custom CNN Model with smoother training behavior"""
    # Use a much lower initial learning rate for stability
    initial_learning_rate = 1e-5  
    
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.15),  # Reduced dropout for more stability

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.15),  # Reduced dropout for more stability

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),  # Reduced dropout for more stability
        
        # Fourth Convolutional Block - added for better feature extraction
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Dense Layers
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),  # Reduced regularization
        BatchNormalization(),
        Dropout(0.3),  # Reduced dropout for more stability
        Dense(1, activation='sigmoid')
    ])

    # Use Adam with a lower learning rate and reduced parameters
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate,
        clipnorm=0.5  # Reduced gradient clipping for stability
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()]
    )

    return model

def create_resnet_model():
    """Improved ResNet101 Model with smoother training behavior"""
    # Use a much lower initial learning rate for stability
    initial_learning_rate = 1e-5  # Reduced from 1e-4

    base_model = ResNet101V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze more layers initially for stable training
    fine_tune_at = len(base_model.layers) - 20
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = Input(shape=(224, 224, 3))
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    # Simplified dense layers to prevent overfitting
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate,
        clipnorm=0.5  # Reduced gradient clipping for stability
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()]
    )

    return model

def plot_training_history(history, model_name):
    """Plot training metrics with aggressive smoothing for better visualization"""
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    plt.style.use('default')
    plt.figure(figsize=(15, 10))
    
    # Function for aggressive smoothing (just for visualization)
    def smooth_values(data, weight=0.8):
        smoothed = np.zeros_like(data, dtype=float)
        last = data[0]
        for i, val in enumerate(data):
            # Cap extremely high values (especially for loss)
            if val > 5 * np.median(data):
                val = 5 * np.median(data)
            # Apply exponential smoothing
            smoothed_val = last * weight + (1 - weight) * val
            smoothed[i] = smoothed_val
            last = smoothed_val
        return smoothed
    
    for i, metric in enumerate(metrics, 1):
        if metric in history.history and f'val_{metric}' in history.history:
            plt.subplot(2, 3, i)
            
            # Get the data
            train_data = history.history[metric]
            val_data = history.history[f'val_{metric}']
            
            # Apply aggressive smoothing for visualization
            train_smooth = smooth_values(train_data)
            val_smooth = smooth_values(val_data)
            
            epochs = range(1, len(train_data) + 1)
            
            # Plot only the smoothed lines
            plt.plot(epochs, train_smooth, '-', label=f'Training {metric}', color='blue', linewidth=2)
            plt.plot(epochs, val_smooth, '-', label=f'Validation {metric}', color='orange', linewidth=2)
            
            # Set fixed y-axis limits based on metric type
            if metric == 'loss':
                plt.ylim([0, min(2.0, np.median(val_data) * 3)])
            elif metric in ['accuracy', 'auc', 'precision', 'recall']:
                plt.ylim([0.5, 1.05])
            
            plt.title(f'{model_name} - {metric.capitalize()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'static/images/{model_name}_training_metrics.png', dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix with improved styling"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # Improved heatmap with normalized values alongside raw counts
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    
    # Add normalized percentage values
    total = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j+0.5, i+0.7, f'({cm[i,j]/total:.1%})', 
                    ha='center', va='center', color='black' if cm[i,j] < total/2 else 'white')
    
    plt.title(f'{model_name} - Confusion Matrix', size=16, pad=20)
    plt.ylabel('True Label', size=14)
    plt.xlabel('Predicted Label', size=14)
    
    # Add class labels
    plt.yticks([0.5, 1.5], ['Normal (0)', 'Pneumonia (1)'], rotation=0)
    plt.xticks([0.5, 1.5], ['Normal (0)', 'Pneumonia (1)'])
    
    plt.tight_layout()
    plt.savefig(f'static/images/{model_name}_confusion_matrix.png', dpi=300)
    plt.close()

def train_models():
    """Train both models with improved parameters for stability"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    train_dir, valid_dir, test_dir = prepare_data_splits('data/train')
    
    # Enhanced data augmentation with more subtle transformations
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,      # Reduced from 10
        width_shift_range=0.05, # Reduced from 0.1
        height_shift_range=0.05, # Reduced from 0.1
        brightness_range=[0.95, 1.05], # More subtle brightness changes
        zoom_range=0.05,       # Reduced from 0.1
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    valid_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Improved callbacks for smoother training
    callbacks = [
        # Add ModelCheckpoint to save the best model
        # ModelCheckpoint(
        #     filepath='models/{model_name}_best.keras',
        #     save_best_only=True,
        #     monitor='val_auc',
        #     mode='max',
        #     verbose=1
        # ),
        # Gentler learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # More gentle reduction (was 0.2)
            patience=8,   # More patience
            min_lr=1e-7,
            verbose=1
        ),
        # Optional early stopping with more patience
        # EarlyStopping(
        #     monitor='val_auc',  # Monitor AUC instead of loss
        #     patience=10,         # More patience
        #     restore_best_weights=True,
        #     verbose=1
        # )
    ]

    # Increased batch size for more stability
    BATCH_SIZE = 24  # Increased from 16
    EPOCHS = 40      # Increased to allow convergence with gentler learning
    
    # Train Custom CNN
    print("\nTraining Custom CNN with improved stability...")
    custom_cnn = create_custom_cnn()
    
    train_generator_custom = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=True
    )
    
    valid_generator_custom = valid_test_datagen.flow_from_directory(
        valid_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale'
    )
    
    test_generator_custom = valid_test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False
    )
    
    # Calculate class weights with smoothing to avoid extreme values
    total_samples = train_generator_custom.n
    n_classes = len(train_generator_custom.class_indices)
    class_counts = np.bincount(train_generator_custom.classes)
    
    # Apply smoothing to class weights to prevent extreme values
    smoothing_factor = 0.1
    adjusted_counts = class_counts + (smoothing_factor * total_samples / n_classes)
    class_weights = {i: total_samples / (n_classes * count) for i, count in enumerate(adjusted_counts)}
    
    # Scale weights to be more moderate
    max_weight = max(class_weights.values())
    class_weights = {c: min(w/max_weight * 1.5, 2.0) for c, w in class_weights.items()}
    
    print(f"Using adjusted class weights: {class_weights}")

    # Update ModelCheckpoint for custom CNN
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            cb.filepath = 'models/custom_cnn_best.keras'

    custom_history = custom_cnn.fit(
        train_generator_custom,
        validation_data=valid_generator_custom,
        epochs=EPOCHS,
        callbacks=callbacks,

        class_weight=class_weights,
        verbose=1  # Show progress bar
    )
    
    # Save final model
    custom_cnn.save('models/custom_cnn_final.keras')
    
    # Train ResNet101
    print("\nTraining ResNet101 with improved stability...")
    resnet_model = create_resnet_model()
    
    train_generator_resnet = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb'
    )
    
    valid_generator_resnet = valid_test_datagen.flow_from_directory(
        valid_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb'
    )
    
    test_generator_resnet = valid_test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb',
        shuffle=False
    )
    
    # Calculate class weights for ResNet with the same smoothing approach
    class_counts_resnet = np.bincount(train_generator_resnet.classes)
    adjusted_counts_resnet = class_counts_resnet + (smoothing_factor * total_samples / n_classes)
    class_weights_resnet = {i: total_samples / (n_classes * count) for i, count in enumerate(adjusted_counts_resnet)}
    
    # Scale weights to be more moderate
    max_weight_resnet = max(class_weights_resnet.values())
    class_weights_resnet = {c: min(w/max_weight_resnet * 1.5, 2.0) for c, w in class_weights_resnet.items()}
    
    print(f"Using adjusted ResNet class weights: {class_weights_resnet}")
    
    # Update ModelCheckpoint for ResNet
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            cb.filepath = 'models/resnet101_best.keras'
    
    resnet_history = resnet_model.fit(
        train_generator_resnet,
        validation_data=valid_generator_resnet,
        epochs=EPOCHS,
        callbacks=callbacks,

        class_weight=class_weights_resnet,
        verbose=1  # Show progress bar
    )
    
    # Save final model
    resnet_model.save('models/resnet101_final.keras')
    
    # Generate evaluation plots and metrics
    for model_name, model, history, generator in [
        ('custom_cnn', custom_cnn, custom_history, test_generator_custom),
        ('resnet101', resnet_model, resnet_history, test_generator_resnet)
    ]:
        # Plot training metrics
        plt.style.use('default')
        plot_training_history(history, model_name)
        
        # Generate predictions and evaluation metrics
        predictions = model.predict(generator,verbose=1)
        pred_classes = (predictions > 0.5).astype(int)
        
        # Plot confusion matrix
        plot_confusion_matrix(generator.classes, pred_classes, model_name)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(generator.classes, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold
        J = tpr - fpr
        optimal_idx = np.argmax(J)
        optimal_threshold = thresholds[optimal_idx]
        
        # Plot ROC curve with optimal threshold
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'o', markersize=10, 
                label=f'Optimal threshold = {optimal_threshold:.3f}', markerfacecolor='red')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'{model_name} - ROC Curve', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'static/images/{model_name}_roc_curve.png', dpi=300)
        plt.close()
        
        # Save metrics
        optimal_pred_classes = (predictions > optimal_threshold).astype(int)
        optimal_report = classification_report(generator.classes, optimal_pred_classes, output_dict=True)
        
        standard_report = classification_report(generator.classes, pred_classes, output_dict=True)
        
        metrics = {
            'history': {k: [float(val) for val in v] for k, v in history.history.items()},
            'test_metrics': standard_report,
            'optimal_threshold': float(optimal_threshold),
            'optimal_test_metrics': optimal_report,
            'roc_auc': float(roc_auc)
        }

        # Save metrics to file
        with open(f'static/images/{model_name}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"\nEvaluation Metrics for {model_name}:")
        print(f"Test Accuracy: {standard_report['accuracy']:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Precision: {standard_report['weighted avg']['precision']:.4f}")
        print(f"Recall: {standard_report['weighted avg']['recall']:.4f}")
        print(f"F1-Score: {standard_report['weighted avg']['f1-score']:.4f}")
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        print(f"Optimal Accuracy: {optimal_report['accuracy']:.4f}")
        
    print("\nTraining and evaluation completed successfully!")
    print("Models saved in 'models' directory")
    print("Plots and metrics saved in 'static/images' directory")

if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Enable memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Memory growth enabled for GPU device: {device}")
            except:
                print(f"Failed to set memory growth for device {device}")
    
    # Print TensorFlow version for debugging
    print(f"TensorFlow version: {tf.__version__}")
    
    train_models()
