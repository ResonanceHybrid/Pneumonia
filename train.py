import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout, 
    BatchNormalization, Input, GlobalAveragePooling2D
)
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
    """Improved Custom CNN Model with balanced metrics and better regularization"""
    initial_learning_rate = 1e-4  # Increased from 1e-5

    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Dense Layers
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate,
        clipnorm=1.0  # Add gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.SpecificityAtSensitivity(0.5)]
    )

    return model

def create_resnet_model():
    """Improved ResNet101 Model with balanced metrics"""
    initial_learning_rate = 1e-4  # Increased from 1e-5

    base_model = ResNet101V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze fewer layers for better training
    fine_tune_at = len(base_model.layers) - 30
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = Input(shape=(224, 224, 3))
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.SpecificityAtSensitivity(0.5)]
    )

    return model

def plot_training_history(history, model_name):
    """Plot and save training metrics"""
    try:
        metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
        plt.style.use('default')
        plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(metrics, 1):
            if metric in history.history and f'val_{metric}' in history.history:
                plt.subplot(2, 3, i)
                plt.plot(history.history[metric], label=f'Training {metric}')
                plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
                plt.title(f'{model_name} - {metric.capitalize()}')
                plt.xlabel('Epoch')
                plt.ylabel(metric.capitalize())
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'static/images/{model_name}_training_metrics.png')
        plt.close()
    except Exception as e:
        print(f"Error plotting training history for {model_name}: {str(e)}")
        plt.close()  # Ensure figure is closed even if error occurs

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'static/images/{model_name}_confusion_matrix.png')
    plt.close()

def train_models():
    """Train both models with improved parameters"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    train_dir, valid_dir, test_dir = prepare_data_splits('data/train')
    
    # Enhanced data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=[0.8, 1.2],
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    valid_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Improved callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    BATCH_SIZE = 16  # Reduced from 32
    EPOCHS = 30
    
    # Train Custom CNN
    print("\nTraining Custom CNN...")
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
    
    # Calculate class weights
    total_samples = train_generator_custom.n
    n_classes = len(train_generator_custom.class_indices)
    class_counts = np.bincount(train_generator_custom.classes)
    class_weights = {i: total_samples / (n_classes * count) for i, count in enumerate(class_counts)}

    custom_history = custom_cnn.fit(
        train_generator_custom,
        validation_data=valid_generator_custom,
        epochs=EPOCHS,
        callbacks=callbacks,
        workers=1,
        use_multiprocessing=False,
        class_weight=class_weights,
        verbose=1  # Show progress bar
    )
    
    custom_cnn.save('models/custom_cnn_final.keras')
    
    # Train ResNet101
    print("\nTraining ResNet101...")
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
    
    # Calculate class weights for ResNet
    class_counts_resnet = np.bincount(train_generator_resnet.classes)
    class_weights_resnet = {i: total_samples / (n_classes * count) for i, count in enumerate(class_counts_resnet)}
    
    resnet_history = resnet_model.fit(
        train_generator_resnet,
        validation_data=valid_generator_resnet,
        epochs=EPOCHS,
        callbacks=callbacks,
        workers=1,
        use_multiprocessing=False,
        class_weight=class_weights_resnet,
        verbose=1  # Show progress bar
    )
    
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
        predictions = model.predict(generator, workers=1, use_multiprocessing=False, verbose=1)
        pred_classes = (predictions > 0.5).astype(int)
        
        # Plot confusion matrix
        plot_confusion_matrix(generator.classes, pred_classes, model_name)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(generator.classes, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'static/images/{model_name}_roc_curve.png')
        plt.close()
        
        # Save metrics
        report = classification_report(generator.classes, pred_classes, output_dict=True)
        metrics = {
            'history': {k: [float(val) for val in v] for k, v in history.history.items()},
            'test_metrics': report,
            'roc_auc': float(roc_auc)
        }

        # Save metrics to file
        with open(f'static/images/{model_name}_metrics.json', 'w') as f:
            json.dump(metrics, f)
            
        print(f"\nEvaluation Metrics for {model_name}:")
        print("Test Accuracy:", report['accuracy'])
        print("ROC AUC:", roc_auc)
        print("Precision:", report['weighted avg']['precision'])
        print("Recall:", report['weighted avg']['recall'])
        print("F1-Score:", report['weighted avg']['f1-score'])
        
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
            tf.config.experimental.set_memory_growth(device, True)
    
    train_models()    