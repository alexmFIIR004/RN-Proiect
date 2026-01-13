import os
import argparse
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import json

# Ensure path context
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.neural_network import config
from src.neural_network.data_loader import MultiModalDataLoader

def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")

def evaluate(model_path):
    # 1. Load Data
    loader = MultiModalDataLoader(base_dir=os.path.join(project_root, 'data'))
    
    (X_test_imu, X_test_img), y_test = loader.load_data('test')
    
    print(f"Test samples: {len(y_test)}")
    
    # 2. Load Model
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Predict
    print("Running inference on test set...")
    y_pred_probs = model.predict([X_test_imu, X_test_img])
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 4. Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    
    print("\n--- Test Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    
    metrics = {
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1),
        "test_precision_macro": float(precision),
        "test_recall_macro": float(recall)
    }
    
    # Save metrics
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {os.path.join(results_dir, 'test_metrics.json')}")
    
    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(project_root, 'docs', 'confusion_matrix.png')
    plot_confusion_matrix(cm, config.CLASS_NAMES, cm_path)
    
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=config.CLASS_NAMES))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=os.path.join(project_root, 'models', 'trained_model.h5'))
    
    args = parser.parse_args()
    
    evaluate(model_path=args.model)
