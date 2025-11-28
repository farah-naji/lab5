import argparse
import os
import time
import pandas as pd
from skimage import io, color
from skimage.filters import gaussian, sobel, gabor, hessian, prewitt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import graycomatrix, graycoprops
from multiprocessing import Pool
import numpy as np
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
parser.add_argument("--features_data", type=str)
args = parser.parse_args()

logger.info(f"Input data path: {args.input_data}")
try:
    logger.info(f"Contents of input_data: {os.listdir(args.input_data)}")
except Exception as e:
    logger.error(f"Error listing input_data: {e}")

start_time = time.time()

def extract_features_for_image(image_path, label):
    img = color.rgb2gray(io.imread(image_path))
    image_id = os.path.basename(image_path)
    features = {}

    filters = {
        'entropy': entropy((img * 255).astype(np.uint8), disk(5)),
        'gaussian': gaussian(img),
        'sobel': sobel(img),
        'gabor': gabor(img)[0],
        'hessian': hessian(img),
        'prewitt': prewitt(img)
    }

    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    props = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    for filter_name, filtered_img in filters.items():
        if filtered_img.min() == filtered_img.max():
            normalized = np.zeros_like(filtered_img, dtype=np.uint8)
        else:
            normalized = ((filtered_img - filtered_img.min()) / (filtered_img.max() - filtered_img.min()) * 255).astype(np.uint8)
        
        glcm = graycomatrix(normalized, [1], angles, symmetric=True, normed=True)
        for prop in props:
            vals = graycoprops(glcm, prop)[0]
            for i, angle in enumerate(angles):
                features[f'{filter_name}_{prop}_{int(np.degrees(angle))}'] = vals[i]

    features['image_id'] = image_id
    features['label'] = label
    return features

image_paths = []
for folder in ['yes', 'no']:
    full_folder = os.path.join(args.input_data, folder)
    logger.info(f"Processing folder: {full_folder}")
    try:
        files = os.listdir(full_folder)
        logger.info(f"Files in {folder}: {len(files)}")
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Filter for images
                image_paths.append((os.path.join(full_folder, file), 1 if folder == 'yes' else 0))
    except Exception as e:
        logger.error(f"Error in folder {folder}: {e}")

logger.info(f"Number of images found: {len(image_paths)}")
if len(image_paths) == 0:
    raise ValueError("No images found in input data - check data asset path/structure")

with Pool(2) as p:
    results = p.starmap(extract_features_for_image, image_paths)

df = pd.DataFrame(results)
logger.info("Writing Parquet file...")
df.to_parquet(args.features_data, index=False)
logger.info("Parquet written successfully.")

extraction_time = time.time() - start_time
mlflow.log_metric("num_images", len(image_paths))
mlflow.log_metric("num_features", len(df.columns) - 2 if not df.empty else 0)
mlflow.log_metric("extraction_time_seconds", extraction_time)
mlflow.log_param("compute_sku", os.environ.get("AZUREML_COMPUTE_SKU", "Standard_DS11_v2"))
print(f"Extraction time: {extraction_time} seconds")
