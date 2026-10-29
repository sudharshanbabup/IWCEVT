"""
download_data.py -- Downloads the UNSW-NB15 and CIC-IDS2017 benchmark datasets
from public Hugging Face mirrors.

Usage:
    python download_data.py [data_dir]
"""
import os
import sys
import ssl
import urllib.request

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

UNSW_FILES = {
    "UNSW_NB15_training-set.csv": "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/test.csv",
    "UNSW_NB15_testing-set.csv": "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/train.csv",
}

CIC_PARQUETS = [
    "Friday-WorkingHours-Afternoon-DDos",
    "Friday-WorkingHours-Afternoon-PortScan",
    "Friday-WorkingHours-Morning",
    "Monday-WorkingHours",
    "Thursday-WorkingHours-Afternoon-Infilteration",
    "Thursday-WorkingHours-Morning-WebAttacks",
    "Tuesday-WorkingHours",
    "Wednesday-workingHours",
]

CIC_BASE = "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning"


def _get_ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def download_file(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  [Already exists] {os.path.basename(dest_path)}")
        return
    print(f"  Downloading {url}\n  -> {dest_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    ctx = _get_ssl_context()
    with urllib.request.urlopen(req, context=ctx) as resp, open(dest_path, "wb") as f:
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunk
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = downloaded / total * 100
                sys.stdout.write(f"\r  Progress: {downloaded / (1024 * 1024):.1f} MB / {total / (1024 * 1024):.1f} MB ({percent:.1f}%)")
                sys.stdout.flush()
        print()


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    unsw_dir = os.path.join(data_dir, "unsw")
    cic_dir = os.path.join(data_dir, "cic")
    os.makedirs(unsw_dir, exist_ok=True)
    os.makedirs(cic_dir, exist_ok=True)

    print(f"Output directory: {os.path.abspath(data_dir)}")
    print("\n[1/2] Fetching UNSW-NB15 files...")
    for filename, url in UNSW_FILES.items():
        download_file(url, os.path.join(unsw_dir, filename))

    print("\n[2/2] Fetching CIC-IDS2017 flow files...")
    for f in CIC_PARQUETS:
        url = f"{CIC_BASE}/{f}.pcap_ISCX.csv.parquet"
        download_file(url, os.path.join(cic_dir, f"{f}.parquet"))

    print("\nAll datasets downloaded successfully.")


if __name__ == "__main__":
    main()
