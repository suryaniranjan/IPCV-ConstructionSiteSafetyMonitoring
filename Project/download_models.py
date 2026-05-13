"""
download_models.py
==================
Downloads MobileNet-SSD model files into the models/ directory.
Run once before starting the safety monitor:

    python download_models.py
"""

import os
import urllib.request

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FILES = {
    "MobileNetSSD_deploy.prototxt": (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
        "master/MobileNetSSD_deploy.prototxt"
    ),
    "MobileNetSSD_deploy.caffemodel": (
        "https://github.com/djmv/MobilNet_SSD_opencv/raw/master/"
        "MobileNetSSD_deploy.caffemodel"
    ),
}


def download(name, url):
    dest = os.path.join(MODELS_DIR, name)
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        print(f"[SKIP] {name} already exists ({size:,} bytes)")
        return
    print(f"[DOWN] Downloading {name} …")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\n[OK]   Saved to {dest}")
    except Exception as e:
        print(f"\n[FAIL] Could not download {name}: {e}")
        print(f"       Please download manually from:\n       {url}")


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = int(pct / 4)
        print(f"\r  [{'█'*bar}{'░'*(25-bar)}] {pct:5.1f}%", end="", flush=True)


if __name__ == "__main__":
    for name, url in FILES.items():
        download(name, url)
    print("\n[INFO] All model files ready.\n")
