"""
Phase 4: Transfer Learning
==========================

Fine-tunes pre-trained FaultFormer on Paderborn real-world damage dataset.
Compares transfer learning vs training from scratch.

Usage:
    python scripts/phase4_transfer_learning.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.pipeline import BearingDatasetPipeline
from src.models.faultformer import build_faultformer

# ==========================================
# 1. PREPARE PADERBORN "REAL DAMAGE" DATA
# ==========================================
def get_paderborn_data():
    """
    Loads Paderborn 'Real Damage' dataset.
    Classes: 
        0: Healthy (K001)
        1: Inner Race - Real Damage (KI04)
        2: Outer Race - Real Damage (KA04)
    """
    pipeline = BearingDatasetPipeline(window_size=2048, stride=1024)
    
    # We use the codes available in your loader.
    # In a full project, you would add more codes like KI14, KA15, etc.
    bearing_codes = {
        'K001': 0,  # Healthy
        'KI04': 1,  # Real Inner Race Fatigue
        'KA04': 2   # Real Outer Race Pitting
    }
    
    X = []
    y = []
    
    print("\n[Phase 4] Loading Paderborn Real Damage Data...")
    for code, label in bearing_codes.items():
        print(f"Processing {code}...")
        try:
            # automatic download -> extract -> resample -> segment
            signals = pipeline.process_paderborn(code)
            if signals is not None:
                X.append(signals)
                # Create label array matching the number of signals
                y.append(np.full(len(signals), label))
        except Exception as e:
            print(f"Error loading {code}: {e}")
    
    if not X:
        raise ValueError("No data was loaded! Check your internet connection and try again. "
                        "The Paderborn dataset requires downloading ~500MB of data.")
            
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    
    # Add channel dimension: (N, 2048) -> (N, 2048, 1)
    X = np.expand_dims(X, axis=-1)
    
    # Split into Train (80%) and Test (20%)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 2. MODEL A: TRANSFER LEARNING
# ==========================================
def create_transfer_model(input_shape):
    print("\n[Model A] Creating Transfer Learning Model...")
    
    # 1. Build the Architecture (Classification Mode)
    model = build_faultformer(input_shape=input_shape, num_classes=3, pretraining=False)
    
    # 2. Load Weights from Phase 3
    try:
        model.load_weights("weights/faultformer_pretrained.weights.h5", skip_mismatch=True)
        print("✅ Pre-trained CWRU weights loaded successfully!")
    except (OSError, ValueError) as e:
        print(f"❌ Pre-trained weights not found or incompatible! Run Phase 3 first. Error: {e}")
        return None

    # 3. Freeze the "Ear" (CNN Tokenizer)
    # We trust the features learned from CWRU, so we don't retrain the tokenizer.
    model.get_layer("CNN_Tokenizer").trainable = False
    print("🔒 CNN Tokenizer layer frozen.")
    
    # (Optional) Freeze Encoder Layers to only train the classifier
    # for layer in model.layers:
    #     if "TransformerEncoder" in layer.name:
    #         layer.trainable = False
            
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==========================================
# 3. MODEL B: TRAINED FROM SCRATCH
# ==========================================
def create_scratch_model(input_shape):
    print("\n Creating Baseline Model (From Scratch)...")
    model = build_faultformer(input_shape=input_shape, num_classes=3, pretraining=False)
    
    # No weight loading, random initialization
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==========================================
# 4. EXECUTION & COMPARISON
# ==========================================
if __name__ == "__main__":
    # 1. Get Data
    X_train, X_test, y_train, y_test = get_paderborn_data()
    print(f"Data Shapes -> Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 2. Train Transfer Model
    transfer_model = create_transfer_model(X_train.shape[1:])
    print("\n--- Training Transfer Model (Pre-trained) ---")
    hist_transfer = transfer_model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test),
        epochs=15, 
        batch_size=32
    )
    
    # 3. Train Scratch Model
    scratch_model = create_scratch_model(X_train.shape[1:])
    print("\n--- Training Scratch Model (Baseline) ---")
    hist_scratch = scratch_model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test),
        epochs=15, 
        batch_size=32
    )
    
    # 4. Visualization of Results
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(hist_transfer.history['val_accuracy'], label='Transfer Learning', marker='o')
    plt.plot(hist_scratch.history['val_accuracy'], label='From Scratch', marker='x', linestyle='--')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(hist_transfer.history['val_loss'], label='Transfer Learning', marker='o')
    plt.plot(hist_scratch.history['val_loss'], label='From Scratch', marker='x', linestyle='--')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    plt.savefig('assets/phase4_comparison_result.png')
    print("\n✅ Comparison plot saved as 'assets/phase4_comparison_result.png'")
    
    # 5. Final Evaluation (Confusion Matrix for Transfer Model)
    print("\n")
    y_pred = np.argmax(transfer_model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Inner', 'Outer']))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Inner', 'Outer'], 
                yticklabels=['Healthy', 'Inner', 'Outer'])
    plt.title('Confusion Matrix (Transfer Model)')
    plt.savefig('assets/phase4_confusion_matrix.png')
    
    # Save final weights for Phase 5 dashboard
    os.makedirs("weights", exist_ok=True)
    transfer_model.save_weights("weights/phase4_final.weights.h5")
    print("\n💾 Final transfer model weights saved to 'weights/phase4_final.weights.h5'")