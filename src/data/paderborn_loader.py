"""
Paderborn University Bearing Dataset Loader
============================================

Downloads and parses the Paderborn bearing dataset with real fatigue damage.
Requires unar for RAR extraction on macOS.

Dataset Info:
    - Source: Paderborn University
    - Sampling Rate: 64 kHz
    - Damage Types: Real fatigue (pitting, spalling)
    - Bearings: K001 (Healthy), KA04 (Outer Race), KI04 (Inner Race)
    
Install unar (macOS):
    brew install unar
"""

import scipy.io as spio
import numpy as np
import os
import requests
import subprocess
import platform
import shutil
import time

def recursive_mat_struct_parse(elem):
    """Recursively converts MATLAB structs to Python dicts."""
    if isinstance(elem, (str, int, float, complex)): return elem
    if np.isscalar(elem): return elem.item()
    if isinstance(elem, np.ndarray):
        if elem.dtype.names:
            if elem.size == 1:
                obj = elem.item()
                return {n: recursive_mat_struct_parse(obj[n]) for n in elem.dtype.names}
        if elem.size == 1 and elem.dtype == 'O':
             return recursive_mat_struct_parse(elem.item())
    return elem

class PaderbornLoader:
    # Official Zenodo Links
    ZENODO_URLS = {
        'K001': 'https://zenodo.org/records/15845309/files/K001.rar?download=1',
        'KA04': 'https://zenodo.org/records/15845309/files/KA04.rar?download=1',
        'KI04': 'https://zenodo.org/records/15845309/files/KI04.rar?download=1'
    }

    def __init__(self, download_dir='./data/paderborn'):
        self.download_dir = os.path.abspath(download_dir)
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
        self._configure_unrar()

    def _configure_unrar(self):
        """Auto-detects unar path for Mac/Linux"""
        self.unar_path = None
        if platform.system() == 'Darwin':
            # Common Homebrew paths for unar
            paths = ['/opt/homebrew/bin/unar', '/usr/local/bin/unar', '/usr/bin/unar']
            for p in paths:
                if os.path.exists(p):
                    self.unar_path = p
                    return

    def _download_file(self, url, save_path):
        """Robust download with progress bar and integrity check."""
        print(f" [PU] Downloading to {os.path.basename(save_path)}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                
                with open(save_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Simple progress indicator
                        if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:
                            print(f"      Progress: {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB", end='\r')
            print("\n [PU] Download complete.")
            return True
        except Exception as e:
            print(f"\n [PU] Download failed: {e}")
            if os.path.exists(save_path):
                os.remove(save_path) # Clean up partial file
            return False

    def download_and_extract(self, bearing_code):
        if bearing_code not in self.ZENODO_URLS:
            print(f"Code {bearing_code} not found.")
            return None
            
        rar_path = os.path.join(self.download_dir, f"{bearing_code}.rar")
        extract_path = os.path.join(self.download_dir, bearing_code)
        
        # 1. Check if we need to extract
        # If extraction folder exists and is not empty, assume done
        if os.path.exists(extract_path) and os.listdir(extract_path):
            # Find the.mat file
            for root, _, files in os.walk(extract_path):
                for file in files:
                    if file.endswith('.mat') and not file.startswith('.'):
                        return os.path.join(root, file)

        # 2. Download if RAR is missing
        if not os.path.exists(rar_path):
            success = self._download_file(self.ZENODO_URLS[bearing_code], rar_path)
            if not success: return None

        # 3. Extract (with Error Handling for Corruption)
        print(f" [PU] Extracting {bearing_code}...")
        try:
            if not os.path.exists(extract_path):
                os.makedirs(extract_path)
            # Use unar command line tool instead of rarfile
            result = subprocess.run(
                [self.unar_path or 'unar', '-o', extract_path, '-f', rar_path],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise Exception(f"unar failed: {result.stderr}")
        except Exception as e:
            print(f" [PU] Extraction Failed: {e}")
            print(f" [PU] The file {rar_path} seems corrupted. Deleting it...")
            os.remove(rar_path) # AUTO-DELETE CORRUPT FILE
            if os.path.exists(extract_path):
                shutil.rmtree(extract_path)
            print(" [PU] Please run the script again to re-download.")
            return None
            
        # 4. Return.mat path
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.mat') and not file.startswith('.'):
                    return os.path.join(root, file)
        
        return None

    def get_all_mat_files(self, bearing_code):
        """Returns ALL .mat files for a bearing code (not just the first one)."""
        # First ensure download/extraction is done
        first_file = self.download_and_extract(bearing_code)
        if not first_file:
            return []
        
        extract_path = os.path.join(self.download_dir, bearing_code)
        mat_files = []
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.mat') and not file.startswith('.'):
                    mat_files.append(os.path.join(root, file))
        return mat_files

    def extract_vibration_data(self, filepath):
        try:
            # struct_as_record=True preserves names
            mat = spio.loadmat(filepath, struct_as_record=True, squeeze_me=False)
        except Exception as e:
            print(f" [PU] Mat read error: {e}")
            return None

        root_keys = [k for k in mat.keys() if not k.startswith('__')]
        if not root_keys: return None
        
        # Access the data structure: mat[root_key][0, 0] gives us the struct
        data = mat[root_keys[0]][0, 0]
        
        # Access Y field which contains all sensors
        if 'Y' not in data.dtype.names:
            return None
        
        y_data = data['Y']
        
        # Y has shape (1, N) where N is number of sensors
        # Find the vibration sensor (usually index 6 or named 'vibration_1')
        for i in range(y_data.shape[1]):
            sensor = y_data[0, i]
            name = str(sensor['Name'])
            if 'vibration' in name.lower():
                # Data shape is (1, samples), flatten it
                return sensor['Data'].flatten()
        
        return None