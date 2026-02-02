"""
Download pretrained models for GPT-SoVITS
"""
import os
import urllib.request
import zipfile
from pathlib import Path

def download_file(url, output_path):
    """Download a file from URL to output_path"""
    print(f"Downloading from {url}...")
    print(f"Saving to {output_path}...")

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download with progress
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgress: {percent}%", end='')

    urllib.request.urlretrieve(url, output_path, reporthook)
    print(f"\nDownload complete: {output_path}")

def unzip_file(zip_path, extract_to):
    """Unzip a file to specified directory"""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete: {extract_to}")

    # Remove zip file
    os.remove(zip_path)
    print(f"Removed zip file: {zip_path}")

def main():
    base_url = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"

    models = {
        "pretrained_models.zip": "GPT_SoVITS",
        "G2PWModel.zip": "GPT_SoVITS/text",
    }

    for filename, extract_to in models.items():
        url = f"{base_url}/{filename}"
        output_path = Path(filename)

        # Check if already downloaded
        if filename == "pretrained_models.zip":
            check_path = Path("GPT_SoVITS/pretrained_models/sv")
        elif filename == "G2PWModel.zip":
            check_path = Path("GPT_SoVITS/text/G2PWModel")
        else:
            check_path = None

        if check_path and check_path.exists():
            print(f"Skipping {filename} - already exists")
            continue

        # Download and extract
        download_file(url, output_path)
        unzip_file(output_path, extract_to)

    print("\n=== All models downloaded successfully! ===")

if __name__ == "__main__":
    main()