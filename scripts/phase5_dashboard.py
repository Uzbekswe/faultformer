"""
Phase 5: Interactive Diagnosis Dashboard
========================================

Gradio-based web interface for real-time bearing fault diagnosis.
Supports file upload, inference, and saliency map visualization.

Usage:
    python scripts/phase5_dashboard.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
import scipy.io
import scipy.signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.pipeline import BearingDatasetPipeline
from src.models.faultformer import build_faultformer

# ==========================================
# 1. CONFIGURATION & MODEL LOADING
# ==========================================
CLASS_NAMES = {0: 'Healthy', 1: 'Inner Race Fault (Real)', 2: 'Outer Race Fault (Real)'}
SEQ_LEN = 2048

def load_trained_model():
    """
    Rebuilds the model and loads the weights trained in Phase 4.
    """
    print("Loading Model...")
    model = build_faultformer(input_shape=(SEQ_LEN, 1), num_classes=3, pretraining=False)
    
    try:
        model.load_weights("weights/phase4_final.weights.h5")
        print("✅ Phase 4 Weights Loaded!")
    except:
        print("⚠️ Weights file not found. Loading Pre-trained weights as fallback.")
        try:
            model.load_weights("weights/faultformer_pretrained.weights.h5", skip_mismatch=True)
        except:
            print("❌ No weights found. Model is untrained.")
            
    return model

# Global Model Instance
MODEL = load_trained_model()

# ==========================================
# 2. T-SNE CLUSTERING VISUALIZATION
# ==========================================
def generate_tsne_plot(X_test, y_test):
    """
    Extracts features from the layer BEFORE classification and plots T-SNE.
    """
    print("Generating T-SNE Plot... this may take a moment.")
    
    # 1. Create a Feature Extractor Model
    # We want the output of 'global_average_pooling1d' (flattened features)
    # Finding the layer name dynamically:
    layer_name = 'global_average_pooling1d'
    feature_model = keras.Model(inputs=MODEL.input, outputs=MODEL.get_layer(layer_name).output)
    
    # 2. Get Features
    features = feature_model.predict(X_test, verbose=0)
    
    # 3. Compute T-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # 4. Plot
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        x=features_2d[:, 0], y=features_2d[:, 1],
        hue=[CLASS_NAMES[y] for y in y_test],
        palette="viridis", s=60, alpha=0.8
    )
    plt.title("T-SNE Visualization of Bearing Fault Features (Phase 5)")
    plt.xlabel("T-SNE Dimension 1")
    plt.ylabel("T-SNE Dimension 2")
    plt.legend(title='Fault Class')
    return plt.gcf()

# ==========================================
# 3. GRADIO UTILITIES (PREPROCESSING)
# ==========================================
def preprocess_uploaded_mat(file_obj):
    """
    Reads a uploaded .mat file, resamples, and windows it.
    Supports both CWRU and Paderborn formats.
    """
    try:
        # Load Mat file
        mat = scipy.io.loadmat(file_obj.name)
        
        # Detect file type and extract signal
        signal = None
        original_sr = 12000  # Default
        
        # Check for CWRU format (keys like 'X105_DE_time', 'X097_DE_time', etc.)
        cwru_keys = [k for k in mat.keys() if '_DE_time' in k or '_FE_time' in k]
        if cwru_keys:
            # CWRU file detected
            signal = mat[cwru_keys[0]].flatten()
            # CWRU files are either 12kHz or 48kHz based on filename
            if '48' in file_obj.name or any('48' in k for k in mat.keys()):
                original_sr = 48000
            else:
                original_sr = 12000
            print(f"CWRU file detected, SR: {original_sr}Hz")
        else:
            # Try Paderborn format or generic
            for key, val in mat.items():
                if isinstance(val, np.ndarray) and not key.startswith('__'):
                    if val.size > 1000:  # Likely signal data
                        if val.ndim == 2: 
                            val = val.flatten()
                        signal = val
                        original_sr = 64000  # Paderborn default
                        break
                
        if signal is None:
            return None, "Could not find signal data in .mat file"

        # Resample to 12kHz target
        target_sr = 12000
        if original_sr != target_sr:
            num_samples = int(len(signal) * target_sr / original_sr)
            signal_resampled = scipy.signal.resample(signal, num_samples)
        else:
            signal_resampled = signal
        
        # Normalize
        signal_norm = (signal_resampled - np.mean(signal_resampled)) / (np.std(signal_resampled) + 1e-6)
        
        # Windowing (Just take the first few windows for quick demo)
        windows = []
        stride = 1024
        for i in range(0, len(signal_norm) - SEQ_LEN, stride):
            windows.append(signal_norm[i:i+SEQ_LEN])
            if len(windows) >= 10: break # Limit to 10 windows for speed
            
        return np.array(windows), "Success"
        
    except Exception as e:
        return None, str(e)

def compute_saliency_map(model, input_window):
    """
    Computes a Saliency Map (Gradient of Output w.r.t Input).
    This highlights which spikes the model used to make the decision.
    """
    input_tensor = tf.convert_to_tensor([input_window], dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor)
        # Get the class with highest probability (axis=1 for batch dimension)
        top_class = tf.argmax(predictions[0])  # Get scalar index for single sample
        top_score = predictions[0, top_class]
        
    # Get gradient of the top predicted class with respect to the input image
    grads = tape.gradient(top_score, input_tensor)
    
    # Absolute value of gradients acts as "Importance"
    grad_abs = tf.abs(grads).numpy()
    
    # Normalize for visualization (0 to 1)
    grad_norm = (grad_abs - np.min(grad_abs)) / (np.max(grad_abs) - np.min(grad_abs) + 1e-8)
    return grad_norm

# ==========================================
# 4. DASHBOARD LOGIC
# ==========================================
def diagnose_bearing(file_obj):
    if file_obj is None:
        return "Please upload a file.", None
    
    # 1. Process File
    windows, status = preprocess_uploaded_mat(file_obj)
    if windows is None:
        return f"Error: {status}", None
    
    # 2. Add Channel Dim
    X_batch = np.expand_dims(windows, axis=-1)
    
    # 3. Predict
    preds = MODEL.predict(X_batch, verbose=0)
    avg_pred = np.mean(preds, axis=0) # Average probabilities across windows
    class_idx = np.argmax(avg_pred)
    confidence = avg_pred[class_idx]
    
    result_str = f"Diagnosis: {CLASS_NAMES[class_idx]}\nConfidence: {confidence:.2%}"
    
    # 4. Generate Attention/Saliency Visualization
    # Use the first window for visualization
    sample_window = X_batch[0]  # Get first window: (2048, 1)
    saliency = compute_saliency_map(MODEL, sample_window)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 4))
    signal_flat = sample_window.flatten()
    ax.plot(signal_flat, label='Vibration Signal', color='black', alpha=0.6, linewidth=1)
    
    # Highlight high importance regions
    # We overlay the saliency as a heatmap color or a secondary line
    # Normalize saliency for colormap
    ax.scatter(np.arange(SEQ_LEN), signal_flat, c=saliency.flatten(), cmap='jet', s=10, label='Model Attention (Saliency)')
    
    plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax, label="Importance")
    ax.set_title(f"Model Attention Map (Looking for {CLASS_NAMES[class_idx]} patterns)")
    ax.legend()
    plt.tight_layout()
    
    return result_str, fig

# ==========================================
# 5. MAIN LAUNCHER
# ==========================================
if __name__ == "__main__":
    # Create Gradio Interface
    interface = gr.Interface(
        fn=diagnose_bearing,
        inputs=gr.File(label="Upload Vibration File (.mat)", file_types=[".mat"]),
        outputs=[
            gr.Textbox(label="Diagnosis Result"),
            gr.Plot(label="Saliency Map")
        ],
        title="FaultFormer: Intelligent Bearing Diagnosis Dashboard",
        description="Upload a Paderborn .mat file. The model will diagnose the fault and visualize which parts of the signal triggered the decision.",
        examples=[
            ["data/paderborn/K001/K001/N15_M07_F10_K001_1.mat"], 
            ["data/paderborn/KA04/KA04/N15_M07_F10_KA04_1.mat"]
        ]
    )
    
    print("🚀 Launching Dashboard...")
    interface.launch(share=True) # share=True creates a public link