"""
Download Pre-trained Weights
============================

Downloads FaultFormer model weights from Google Drive.

Usage:
    python scripts/download_weights.py
"""

import os
import requests

# Google Drive direct download links
WEIGHTS = {
    "faultformer_pretrained.weights.h5": {
        "url": "https://drive.google.com/uc?export=download&id=1LHlYDG208ozm45U0vh8nLvmh4DKIbSUR",
        "description": "Phase 3: Self-supervised pretrained weights (CWRU)"
    },
    "phase4_final.weights.h5": {
        "url": "https://drive.google.com/uc?export=download&id=1K9YC3hiw1TLPkMOZ4K-LN0PsHq7PhTe2",
        "description": "Phase 4: Transfer learning weights (Paderborn)"
    }
}


def download_file(url, save_path, description):
    """Download a file from Google Drive."""
    print(f"\n📥 Downloading: {description}")
    print(f"   URL: {url}")
    print(f"   Saving to: {save_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"   Progress: {percent:.1f}%", end='\r')
        
        print(f"\n   ✅ Downloaded successfully! ({os.path.getsize(save_path) / 1024 / 1024:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"\n   ❌ Download failed: {e}")
        return False


def main():
    print("=" * 60)
    print("FaultFormer - Pre-trained Weights Downloader")
    print("=" * 60)
    
    # Create weights directory
    weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    success_count = 0
    for filename, info in WEIGHTS.items():
        save_path = os.path.join(weights_dir, filename)
        
        # Skip if already exists
        if os.path.exists(save_path):
            print(f"\n⏭️  {filename} already exists. Skipping.")
            success_count += 1
            continue
        
        if download_file(info["url"], save_path, info["description"]):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Download complete! {success_count}/{len(WEIGHTS)} files downloaded.")
    print("=" * 60)
    
    if success_count == len(WEIGHTS):
        print("\n🎉 All weights downloaded successfully!")
        print("   You can now run:")
        print("   - python scripts/phase5_dashboard.py  (for inference)")
    else:
        print("\n⚠️  Some downloads failed. Please try again or download manually.")


if __name__ == "__main__":
    main()
