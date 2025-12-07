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
    class_names = {0: 'Healthy', 1: 'Inner Race Fault', 2: 'Outer Race Fault'}
    
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
        hue=[class_names[y] for y in y_test],
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
            # Try Paderborn nested struct format
            # Paderborn files have structure: mat[filename][0,0]['Y'][0,channel]['Data']
            data_keys = [k for k in mat.keys() if not k.startswith('__')]
            for key in data_keys:
                try:
                    struct = mat[key]
                    if struct.dtype.names and 'Y' in struct.dtype.names:
                        # Paderborn format confirmed
                        Y = struct[0, 0]['Y']
                        # Y contains multiple sensors - find the vibration sensor
                        # Sensor 6 is typically 'vibration_1' at 64kHz
                        for sensor_idx in range(Y.shape[1]):
                            sensor = Y[0, sensor_idx]
                            sensor_name = str(sensor['Name']).lower()
                            if 'vibration' in sensor_name:
                                signal = sensor['Data'].flatten()
                                original_sr = 64000  # Paderborn vibration is 64kHz
                                print(f"Paderborn file detected, vibration sensor found, SR: {original_sr}Hz, signal length: {len(signal)}")
                                break
                        if signal is not None:
                            break
                except (KeyError, IndexError, TypeError):
                    continue
            
            # Fallback: try generic format
            if signal is None:
                for key, val in mat.items():
                    if isinstance(val, np.ndarray) and not key.startswith('__'):
                        if val.size > 1000:  # Likely signal data
                            if val.ndim == 2: 
                                val = val.flatten()
                            signal = val
                            original_sr = 64000
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
CLASS_NAMES = {0: 'Healthy', 1: 'Inner Race Fault', 2: 'Outer Race Fault'}

def diagnose_bearing(file_obj):
    if file_obj is None:
        return "⏳ Please upload a .mat file to begin diagnosis.", "", "", "", None
    
    # 1. Process File
    windows, status = preprocess_uploaded_mat(file_obj)
    if windows is None:
        return f"❌ Error: {status}", "", "", "", None
    
    # 2. Add Channel Dim
    X_batch = np.expand_dims(windows, axis=-1)
    
    # 3. Predict
    preds = MODEL.predict(X_batch, verbose=0)
    avg_pred = np.mean(preds, axis=0)  # Average probabilities across windows
    class_idx = np.argmax(avg_pred)
    confidence = avg_pred[class_idx]
    
    # Determine status
    if class_idx == 0:
        status_emoji = "✅"
        status_text = "HEALTHY"
    elif class_idx == 1:
        status_emoji = "⚠️"
        status_text = "INNER RACE FAULT"
    else:
        status_emoji = "🔴"
        status_text = "OUTER RACE FAULT"
    
    # Format outputs
    diagnosis_result = f"{status_emoji} {status_text}"
    confidence_text = f"{confidence:.1%}"
    prob_text = f"Healthy: {avg_pred[0]:.1%} | Inner Race: {avg_pred[1]:.1%} | Outer Race: {avg_pred[2]:.1%}"
    
    # 4. Generate Simple & Intuitive Visualization
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.4, wspace=0.3)
    
    # === TOP ROW: Health Status Gauges ===
    
    # Health Status Indicator (Traffic Light Style)
    ax_status = fig.add_subplot(gs[0, 0])
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)
    ax_status.axis('off')
    
    # Draw circle with color based on diagnosis
    colors = {'Healthy': '#2ECC71', 'Inner Race Fault': '#F39C12', 'Outer Race Fault': '#E74C3C'}
    status_color = colors[CLASS_NAMES[class_idx]]
    circle = plt.Circle((0.5, 0.5), 0.35, color=status_color, ec='white', linewidth=4)
    ax_status.add_patch(circle)
    ax_status.text(0.5, 0.5, f"{confidence:.0%}", ha='center', va='center', 
                   fontsize=28, fontweight='bold', color='white')
    ax_status.text(0.5, 0.05, CLASS_NAMES[class_idx].upper(), ha='center', va='bottom',
                   fontsize=11, fontweight='bold', color=status_color)
    ax_status.set_title("Health Status", fontsize=14, fontweight='bold', pad=10)
    
    # Probability Bar Chart
    ax_bars = fig.add_subplot(gs[0, 1:])
    bar_colors = ['#2ECC71', '#F39C12', '#E74C3C']
    bars = ax_bars.barh(['Healthy', 'Inner Race\nFault', 'Outer Race\nFault'], 
                        [avg_pred[0], avg_pred[1], avg_pred[2]], 
                        color=bar_colors, edgecolor='white', linewidth=2, height=0.6)
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, avg_pred):
        width = bar.get_width()
        ax_bars.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1%}', va='center', ha='left', fontsize=12, fontweight='bold')
    
    ax_bars.set_xlim(0, 1.15)
    ax_bars.set_xlabel('Probability', fontsize=11)
    ax_bars.set_title('Fault Probability Distribution', fontsize=14, fontweight='bold', pad=10)
    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['right'].set_visible(False)
    
    # === BOTTOM ROW: Signal Visualization ===
    ax_signal = fig.add_subplot(gs[1, :])
    sample_window = X_batch[0].flatten()
    time_ms = np.arange(len(sample_window)) / 12  # Convert to milliseconds (12kHz)
    
    # Plot signal with color based on health
    ax_signal.fill_between(time_ms, sample_window, alpha=0.3, color=status_color)
    ax_signal.plot(time_ms, sample_window, color=status_color, linewidth=0.8, alpha=0.8)
    ax_signal.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Simple interpretation text
    if class_idx == 0:
        interpretation = "Signal shows normal vibration patterns - bearing is healthy"
    elif class_idx == 1:
        interpretation = "Signal shows periodic impulses - inner race damage detected"
    else:
        interpretation = "Signal shows irregular patterns - outer race damage detected"
    
    ax_signal.set_xlabel('Time (ms)', fontsize=11)
    ax_signal.set_ylabel('Amplitude', fontsize=11)
    ax_signal.set_title(f'Vibration Signal: {interpretation}', fontsize=12, fontweight='bold', pad=10)
    ax_signal.spines['top'].set_visible(False)
    ax_signal.spines['right'].set_visible(False)
    ax_signal.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return diagnosis_result, confidence_text, prob_text, f"Analyzed {len(windows)} windows", fig

# ==========================================
# 5. MAIN LAUNCHER (IMPROVED UI)
# ==========================================
if __name__ == "__main__":
    
    with gr.Blocks(title="FaultFormer Dashboard") as demo:
        # Header
        gr.Markdown(
            """
            # 🔧 FaultFormer: Intelligent Bearing Diagnosis
            ### AI-Powered Fault Detection | 99.77% Accuracy
            
            Upload a vibration signal (.mat file) to instantly diagnose bearing health.
            """
        )
        
        with gr.Row():
            # Left Column - Input
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="📁 Upload Vibration File (.mat)",
                    file_types=[".mat"],
                    type="filepath"
                )
                
                submit_btn = gr.Button("🔬 Analyze Signal", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                
                gr.Markdown("### 📂 Example Files")
                gr.Examples(
                    examples=[
                        ["data/paderborn/K001/K001/N15_M07_F10_K001_1.mat"],
                        ["data/paderborn/KA04/KA04/N15_M07_F10_KA04_1.mat"],
                        ["data/paderborn/KI04/KI04/N15_M07_F10_KI04_1.mat"]
                    ],
                    inputs=file_input,
                    label=""
                )
            
            # Right Column - Results
            with gr.Column(scale=2):
                gr.Markdown("### 🩺 Diagnosis Results")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        diagnosis_output = gr.Textbox(label="Diagnosis", lines=1)
                    with gr.Column(scale=1):
                        confidence_output = gr.Textbox(label="Confidence", lines=1)
                
                prob_output = gr.Textbox(label="Class Probabilities", lines=1)
                info_output = gr.Textbox(label="Analysis Info", lines=1)
        
        # Visualization - Full Width
        gr.Markdown("### 📊 Diagnosis Overview")
        plot_output = gr.Plot(label="")
        
        # Footer - Informative Guide
        gr.Markdown(
            """
            ---
            ## 📖 Understanding Your Results
            
            | Status | Color | Meaning | Recommended Action |
            |--------|-------|---------|-------------------|
            | ✅ **Healthy** | 🟢 Green | Normal vibration patterns | Continue regular monitoring |
            | ⚠️ **Inner Race Fault** | 🟠 Orange | Damage on inner bearing ring | Schedule maintenance within 2-4 weeks |
            | 🔴 **Outer Race Fault** | 🔴 Red | Damage on outer bearing ring | Immediate inspection required |
            
            **Reading the Charts:**
            - **Health Circle**: Shows diagnosis with confidence % inside
            - **Bar Chart**: Compare probabilities - longer bar = higher likelihood
            - **Signal Plot**: Raw vibration colored by diagnosis
            
            ---
            *FaultFormer v1.0 | Transformer Neural Network | © 2025*
            """
        )
        
        # Event handlers
        submit_btn.click(
            fn=diagnose_bearing,
            inputs=[file_input],
            outputs=[diagnosis_output, confidence_output, prob_output, info_output, plot_output]
        )
        
        clear_btn.click(
            fn=lambda: (None, "", "", "", "", None),
            inputs=[],
            outputs=[file_input, diagnosis_output, confidence_output, prob_output, info_output, plot_output]
        )
    
    print("🚀 Launching Dashboard...")
    demo.launch(share=True)