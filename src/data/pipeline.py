"""
Unified Data Processing Pipeline
================================

Handles data loading, resampling, and preprocessing for both
CWRU and Paderborn bearing datasets.
"""

import os
import numpy as np
import scipy.signal
from sklearn.preprocessing import StandardScaler

from src.data.cwru_loader import CWRUManager
from src.data.paderborn_loader import PaderbornLoader

class BearingDatasetPipeline:
    def __init__(self, target_sr=12000, window_size=2048, stride=1024):
        self.target_sr = target_sr
        self.window_size = window_size
        self.stride = stride
        self.scaler = StandardScaler()
        
    def resample_signal(self, signal, original_sr):
        if original_sr == self.target_sr:
            return signal
        duration = len(signal) / original_sr
        new_num_samples = int(duration * self.target_sr)
        return scipy.signal.resample(signal, new_num_samples)

    def segment_and_normalize(self, signal):
        signal = self.scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        n_samples = len(signal)
        if n_samples < self.window_size:
            return np.array()
            
        n_windows = (n_samples - self.window_size) // self.stride + 1
        windows = np.zeros((n_windows, self.window_size))
        
        for i in range(n_windows):
            start = i * self.stride
            windows[i] = signal[start : start + self.window_size]
            
        return windows

    def process_cwru(self, file_id):
        print(f"\n--- Processing CWRU File {file_id} ---")
        manager = CWRUManager()
        filepath = manager.download_file(file_id)
        
        if not filepath: return None
        
        parsed = manager.load_and_parse(filepath)
        if not parsed: return None
        
        print(f" Original SR: {parsed['sr']}Hz -> Target SR: {self.target_sr}Hz")
        resampled = self.resample_signal(parsed['signal'], parsed['sr'])
        windows = self.segment_and_normalize(resampled)
        print(f" Output Shape: {windows.shape}")
        return windows

    def process_paderborn(self, bearing_code):
        print(f"\n--- Processing Paderborn File {bearing_code} ---")
        loader = PaderbornLoader()
        
        # Get ALL .mat files for this bearing code
        mat_files = loader.get_all_mat_files(bearing_code)
        
        if not mat_files: 
            print(" Skipping Paderborn due to download/extract failure.")
            return None
        
        all_windows = []
        for filepath in mat_files:
            raw_signal = loader.extract_vibration_data(filepath)
            if raw_signal is None: 
                continue
            
            resampled = self.resample_signal(raw_signal, 64000)
            windows = self.segment_and_normalize(resampled)
            all_windows.append(windows)
        
        if not all_windows:
            print(" Failed to extract vibration data from any file.")
            return None
        
        combined = np.concatenate(all_windows, axis=0)
        print(f" Original SR: 64000Hz -> Target SR: {self.target_sr}Hz")
        print(f" Output Shape: {combined.shape}")
        return combined

# --- Main Execution ---
if __name__ == "__main__":
    # Settings matching FaultFormer paper
    pipeline = BearingDatasetPipeline(target_sr=12000, window_size=2048)
    
    # 1. Test CWRU
    pipeline.process_cwru('105')
    
    # 2. Test Paderborn
    # This will now auto-resume if your internet fails
    pipeline.process_paderborn('K001')
    
    print("\n✅ Phase 1 Verification Complete!")