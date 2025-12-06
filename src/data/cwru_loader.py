"""
CWRU Bearing Dataset Manager
============================

Downloads and parses the Case Western Reserve University bearing dataset.
Supports multiple file IDs with different fault types and sampling rates.

Dataset Info:
    - URL: https://engineering.case.edu/bearingdatacenter
    - Sampling Rates: 12kHz (most files), 48kHz (fan-end files)
    - Fault Types: Normal, Inner Race, Ball, Outer Race
"""

import os
import re
import time
import requests
import scipy.io
import numpy as np


class CWRUManager:
    """
    Manager for downloading and parsing CWRU bearing datasets.
    Includes retry logic and robust error handling.
    """
    BASE_URL = "https://engineering.case.edu/sites/default/files/"
    
    # Sampling rate mapping for different file IDs
    SR_MAP = {
        '97': 48000, '98': 48000, '99': 48000, '100': 48000, 
        '105': 12000, '106': 12000, '107': 12000, '108': 12000,
        '118': 12000, '119': 12000, '120': 12000, '121': 12000,
        '130': 12000, '131': 12000, '132': 12000, '133': 12000,
        '169': 12000, '3001': 12000, '3005': 48000 
    }

    def __init__(self, download_dir='./data/cwru'):
        """
        Initialize CWRU Manager.
        
        Args:
            download_dir: Directory to store downloaded .mat files
        """
        self.download_dir = os.path.abspath(download_dir)
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def _get_url_for_id(self, file_id):
        """Generate download URL for a file ID."""
        return f"{self.BASE_URL}{file_id}.mat"

    def download_file(self, file_id):
        """
        Download a CWRU .mat file.
        
        Args:
            file_id: File identifier (e.g., '105', '118')
            
        Returns:
            Path to downloaded file, or None if failed
        """
        filename = f"{file_id}.mat"
        save_path = os.path.join(self.download_dir, filename)
        
        if os.path.exists(save_path):
            print(f" File {filename} exists. Skipping.")
            return save_path
            
        url = self._get_url_for_id(file_id)
        print(f" Downloading {filename}...")
        
        # Download with retry logic
        for attempt in range(3):
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return save_path
            except Exception as e:
                print(f" Attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2)
        
        return None

    def load_and_parse(self, filepath):
        """
        Load and parse a CWRU .mat file.
        
        Args:
            filepath: Path to .mat file
            
        Returns:
            Dictionary with 'signal', 'sr', 'id' keys, or None if failed
        """
        if not filepath or not os.path.exists(filepath):
            return None

        try:
            mat_data = scipy.io.loadmat(filepath)
        except Exception as e:
            print(f"Error reading .mat file: {e}")
            return None

        # Regex to find Drive End data (primary accelerometer)
        de_pattern = re.compile(r"X?\d+_DE_time")
        
        signal = None
        for key in mat_data.keys():
            if de_pattern.search(key):
                signal = mat_data[key].flatten()
                break
        
        if signal is None:
            print(f"Warning: No DE data found in {filepath}")
            return None
            
        filename = os.path.basename(filepath).replace('.mat', '')
        sr = self.SR_MAP.get(filename, 12000)
            
        return {'signal': signal, 'sr': sr, 'id': filename}


if __name__ == "__main__":
    # Test the loader
    manager = CWRUManager()
    filepath = manager.download_file('105')
    if filepath:
        data = manager.load_and_parse(filepath)
        if data:
            print(f"Loaded {data['id']}: {len(data['signal'])} samples @ {data['sr']}Hz")
