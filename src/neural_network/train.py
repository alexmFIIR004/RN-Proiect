import os
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import callbacks

# Ensure path context
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.neural_network import config
from src.neural_network.data_loader import MultiModalDataLoader
from src.neural_network.model import build_floor_classifier_model

def plot_history(history, output_path):
    """
    Plots training history (accuracy and loss) and saves to file.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Loss curve saved to {output_path}")

def train(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, use_early_stopping=True):
    # 1. Load Data
    loader = MultiModalDataLoader(base_dir=os.path.join(project_root, 'data'))
    
    (X_train_imu, X_train_img), y_train = loader.load_data('train')
    (X_val_imu, X_val_img), y_val = loader.load_data('validation')
    
    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    
    # 2. Build Model
    model = build_floor_classifier_model(
        input_shape_imu=config.INPUT_SHAPE_IMU,
        input_shape_img=config.INPUT_SHAPE_IMG,
        num_classes=config.NUM_CLASSES
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 3. Callbacks
    callback_list = []
    
    # Checkpoint (save best only)
    model_save_path = os.path.join(project_root, 'models', 'trained_model.h5')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callback_list.append(callbacks.ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    ))
    
    # CSV Logger
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_logger = callbacks.CSVLogger(os.path.join(results_dir, 'training_history.csv'))
    callback_list.append(csv_logger)
    
    #Here we check if early stopping is enabled and necessary
    if use_early_stopping:
    
        es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        callback_list.append(es)
        
    # Learning Rate Scheduler
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3, 
        min_lr=1e-6,
        verbose=1
    )
    callback_list.append(lr_scheduler)

    # 4. Train
    print("Starting training...")
    history = model.fit(
        x=[X_train_imu, X_train_img],
        y=y_train,
        validation_data=([X_val_imu, X_val_img], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list
    )
    
    # 5. Save final plot
    plot_path = os.path.join(project_root, 'docs', 'loss_curve.png')
    plot_history(history, plot_path)
    
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--early_stopping', action='store_true', default=True, help="Enable early stopping") # Default True for Level 2
    
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, use_early_stopping=args.early_stopping)
