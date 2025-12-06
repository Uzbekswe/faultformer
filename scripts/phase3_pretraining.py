"""
Phase 3: Self-Supervised Pre-training
======================================

Pre-trains FaultFormer on CWRU dataset using Masked Signal Modeling.
The model learns to reconstruct masked portions of vibration signals.

Usage:
    python scripts/phase3_pretraining.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.pipeline import BearingDatasetPipeline
from src.models.faultformer import build_faultformer
from src.models.pretrain_utils import get_pretraining_dataset

# ==========================================
# 1. BULK DATA AGGREGATION
# ==========================================
def gather_cwru_dataset():
    """
    Downloads and processes multiple CWRU files to create a diverse 
    unlabeled dataset for pre-training.
    """
    pipeline = BearingDatasetPipeline(window_size=2048, stride=512)
    
    # List of File IDs representing different conditions
    # 97-100: Normal (0-3 HP)
    # 105-108: Inner Race Fault 0.007"
    # 118-121: Ball Fault 0.007"
    # 130-133: Outer Race Fault 0.007"
    # We use a mix to ensure the model learns all types of physics.
    file_ids = ['97', '98', '99', '105', '106', '118', '119', '130', '131']
    
    all_signals = []
    
    print("gathering CWRU data for pre-training...")
    for fid in file_ids:
        try:
            # We don't care about labels for pre-training, just the signals
            data = pipeline.process_cwru(fid)
            if data is not None:
                all_signals.append(data)
        except Exception as e:
            print(f"Skipping {fid}: {e}")
            
    if not all_signals:
        raise ValueError("No data collected! Check internet or file paths.")
        
    # Concatenate all windows into one big array
    # Shape: (Total_Samples, 2048)
    dataset = np.concatenate(all_signals, axis=0)
    
    # Add channel dimension for CNN: (N, 2048, 1)
    dataset = np.expand_dims(dataset, axis=-1)
    
    print(f"✅ Total Pre-training Samples: {dataset.shape}")
    return dataset

# ==========================================
# 2. VISUALIZATION UTILITY
# ==========================================
def plot_reconstruction(model, sample_signal):
    """
    Runs the model on a sample and plots Original vs Reconstructed.
    This visually proves the model understands the physics.
    """
    # 1. Prepare sample
    sample = np.expand_dims(sample_signal, axis=0) # (1, 2048, 1)
    
    # 2. Mask it manually for visualization (75% mask)
    # We just zero out random chunks to simulate what the model sees
    mask_indices = np.random.choice(2048, size=int(2048*0.75), replace=False)
    masked_sample = sample.copy()
    masked_sample[:, mask_indices, :] = 0
    
    # 3. Predict
    reconstruction = model.predict(masked_sample, verbose=0)
    
    # 4. Plot - squeeze to 1D for plotting
    plt.figure(figsize=(15, 5))
    plt.plot(sample.squeeze(), label='Original Signal', color='black', alpha=0.5)
    plt.plot(reconstruction.squeeze(), label='FaultFormer Reconstruction', color='red', linewidth=1)
    plt.title("Self-Supervised Learning: Reconstructing a 75% Masked Signal")
    plt.legend()
    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    plt.savefig('assets/reconstruction_proof.png')
    plt.show()
    print("✅ Reconstruction plot saved as 'assets/reconstruction_proof.png'")

# ==========================================
# 3. MAIN PRE-TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    # --- Step 1: Get Data ---
    X_train = gather_cwru_dataset()
    
    # --- Step 2: Create Self-Supervised Pipeline ---
    # We use a high masking ratio (75%) because vibration signals are redundant.
    # The model must work hard to infer the missing pieces.
    BATCH_SIZE = 64
    train_ds = get_pretraining_dataset(X_train, batch_size=BATCH_SIZE, mask_ratio=0.75)
    
    # --- Step 3: Build PRE-TRAINING Model ---
    # Note: pretraining=True activates the Reconstruction Head (Decoder)
    model = build_faultformer(input_shape=(2048, 1), pretraining=True)
    
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss='mse' # Mean Squared Error: Comparing Reconstructed vs Original
    )
    
    model.summary()
    
    # --- Step 4: Train ---
    print("\n🚀 Starting Self-Supervised Pre-Training...")
    print("Goal: Minimize MSE Loss (Reconstruction Error)")
    
    # We train for 10 epochs (fast demonstration). In production, do 50+.
    history = model.fit(train_ds, epochs=10)
    
    # --- Step 5: Save the "Brain" ---
    # We save the weights so we can transfer them to the Paderborn task later.
    os.makedirs("weights", exist_ok=True)
    model.save_weights("weights/faultformer_pretrained.weights.h5")
    print("\n💾 Pre-trained weights saved to 'weights/faultformer_pretrained.weights.h5'")
    
    # --- Step 6: Verify Physics Learning ---
    # Pick a random sample and see if the model can fix it
    test_sample = X_train[np.random.randint(len(X_train))]
    plot_reconstruction(model, test_sample)