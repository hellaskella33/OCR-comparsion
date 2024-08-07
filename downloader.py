import os
import gdown
import requests
import tarfile
import argparse
from tqdm import tqdm

MODELS = {
    'en_PP-OCRv4_rec_train': {
        'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar',
        'relative_path': 'paddle_models/en_PP-OCRv4_rec_train'
    },
    'en_PP-OCRv3_det_slim_distill_train': {
        'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_distill_train.tar',
        'relative_path': 'paddle_models/en_PP-OCRv3_det_slim_distill_train'
    }
}

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def extract_tar(tar_path, extract_path):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_path)

def download_and_extract_model(model_name, base_path):
    model_info = MODELS[model_name]
    url = model_info['url']
    path = os.path.join(base_path, model_info['relative_path'])
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    tar_filename = os.path.join(os.path.dirname(path), f"{model_name}.tar")
    
    print(f"Downloading {model_name}...")
    download_file(url, tar_filename)
    
    print(f"Extracting {model_name}...")
    extract_tar(tar_filename, os.path.dirname(path))
    
    print(f"Cleaning up...")
    os.remove(tar_filename)
    
    print(f"{model_name} downloaded and extracted successfully.")

def main(args):
    base_path = args.base_path
    for model_name in MODELS:
        download_and_extract_model(model_name, base_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract PaddleOCR models")
    parser.add_argument("--base_path", type=str, default=".", help="Base path for storing the models")
    args = parser.parse_args()
    main(args)