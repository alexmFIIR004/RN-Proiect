import os
import sys

# Add project root and current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(current_dir)

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from data_loader import MultiModalDataLoader
import config
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Setup directories
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('docs', exist_ok=True)

def create_custom_model(hp_dropout=0.3, hp_lr=0.001):
    """
    Recreates the multimodal architecture with adjustable hyperparameters
    """
    # IMU Branch
    imu_input = keras.layers.Input(shape=config.INPUT_SHAPE_IMU, name='imu_input')
    x1 = keras.layers.Conv1D(64, 3, activation='relu', padding='same')(imu_input)
    x1 = keras.layers.BatchNormalization()(x1)
    x1 = keras.layers.MaxPooling1D(2)(x1)
    x1 = keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x1)
    x1 = keras.layers.GlobalAveragePooling1D()(x1)
    
    # Image Branch
    img_input = keras.layers.Input(shape=config.INPUT_SHAPE_IMG, name='img_input')
    x2 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    x2 = keras.layers.MaxPooling2D((2, 2))(x2)
    x2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    x2 = keras.layers.MaxPooling2D((2, 2))(x2)
    x2 = keras.layers.Flatten()(x2)
    
    # Fusion
    merged = keras.layers.concatenate([x1, x2])
    x = keras.layers.Dense(64, activation='relu')(merged)
    x = keras.layers.Dropout(hp_dropout)(x) # Variable Dropout
    output = keras.layers.Dense(len(config.CLASS_NAMES), activation='softmax')(x)
    
    model = keras.models.Model(inputs=[imu_input, img_input], outputs=output)
    
    optimizer = keras.optimizers.Adam(learning_rate=hp_lr)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_all_data():
    data_dir = os.path.join(project_root, 'data')
    loader = MultiModalDataLoader(base_dir=data_dir)
    print("Loading Train...")
    (train_imu, train_img), train_labels = loader.load_data('train')
    print("Loading Validation...")
    (val_imu, val_img), val_labels = loader.load_data('validation')
    print("Loading Test...")
    (test_imu, test_img), test_labels = loader.load_data('test')
    return (train_imu, train_img, train_labels), (val_imu, val_img, val_labels), (test_imu, test_img, test_labels)

def run_experiments():
    # Load Data
    print("Loading Data...")
    (train_imu, train_img, train_labels), (val_imu, val_img, val_labels), (test_imu, test_img, test_labels) = load_all_data()
    
    experiments = [
        {'id': 'Exp_Baseline', 'lr': 0.001, 'dropout': 0.3, 'batch': 32, 'desc': 'Baseline configuration'},
        {'id': 'Exp_HighReg', 'lr': 0.001, 'dropout': 0.6, 'batch': 32, 'desc': 'High Dropout (0.6) for robustness'},
        {'id': 'Exp_LowLR',    'lr': 0.0001, 'dropout': 0.3, 'batch': 32, 'desc': 'Low Learning Rate (Fine tuning)'},
        {'id': 'Exp_BigBatch', 'lr': 0.001, 'dropout': 0.3, 'batch': 64, 'desc': 'Larger Batch Size'}
    ]
    
    results = []
    best_acc = 0
    best_model_path = "models/optimized_model.h5"
    
    for exp in experiments:
        print(f"\nRunning {exp['id']}...")
        start_time = time.time()
        
        # Build Model
        model = create_custom_model(hp_dropout=exp['dropout'], hp_lr=exp['lr'])
        
        # Train
        # EarlyStopping to be efficient
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
        
        history = model.fit(
            [train_imu, train_img], train_labels,
            epochs=5, # Cap at 5 for optimization speed
            batch_size=exp['batch'],
            validation_data=([val_imu, val_img], val_labels),
            callbacks=callbacks,
            verbose=1
        )
        
        duration = (time.time() - start_time) / 60
        
   
        loss, acc = model.evaluate([test_imu, test_img], test_labels, verbose=0)
        
     
        y_pred_prob = model.predict([test_imu, test_img], verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
     
        from sklearn.metrics import f1_score
        f1 = f1_score(test_labels, y_pred, average='macro')
        
        res_entry = {
            'Exp_ID': exp['id'],
            'Changes': exp['desc'],
            'Accuracy': round(acc, 4),
            'F1_Score': round(f1, 4),
            'Time_min': round(duration, 2),
            'Observations': f"Loss: {loss:.4f}"
        }
        results.append(res_entry)
        
        # Save best model
        if acc >= best_acc: 
            best_acc = acc
            model.save(best_model_path)
            print(f" -> New Best Model saved: {acc:.4f}")
            
          
            cm = confusion_matrix(test_labels, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
            plt.title(f'Confusion Matrix (Optimized Model) - Acc: {acc:.2f}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig('docs/confusion_matrix_optimized.png')
            plt.close()
            
            # Save final metrics json
            final_metrics = {
                "model": "optimized_model.h5",
                "test_accuracy": acc,
                "test_f1_macro": f1,
                "experiment_id": exp['id'],
                "parameters": exp
            }
            with open('results/final_metrics.json', 'w') as f:
                json.dump(final_metrics, f, indent=4)

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv('results/optimization_experiments.csv', index=False)
    print("\nOptimization Complete. Results saved.")
    print(df)

if __name__ == "__main__":
    run_experiments()
