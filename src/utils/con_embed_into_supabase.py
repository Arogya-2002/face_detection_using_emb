import json
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
import torch
from src.utils.models import resnet, mtcnn, device
from supabase import create_client, Client, ClientOptions
import time
from src.constants import *
from src.exceptions import CustomException
from src.logger import logging
import sys


# ------------------ Supabase Setup ------------------
try:
    
    options=ClientOptions(schema="dc")
    supabase: Client = create_client(FSUPABASE_URL, FSUPABASE_KEY,options=options)
    logging.info("Connected to Supabase successfully.")
except Exception as e:
    raise CustomException(e,sys)

# ------------------ Track Seen Images ------------------
seen_images = {}

# ------------------ Function: Download Image ------------------
def download_image(image_path):
    try:
        url = FIMAGE_URL_PREFIX + image_path
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        logging.info(f"Downloaded image: {image_path}")
        return img
    except Exception as e:
        logging.error(f"Could not download image {image_path}: {e}")
        raise CustomException(e,sys)

# ------------------ Function: Get Embedding ------------------
def get_embedding(img, face_index):
    try:
        faces = mtcnn(img)
        if faces is None or len(faces) <= face_index:
            logging.warning(f"No face #{face_index + 1} found.")
            return None
        face = faces[face_index]
        if face.ndim == 3:
            face = face.unsqueeze(0)
        emb = resnet(face.to(device)).detach().cpu().numpy()[0]
        logging.info(f"Generated embedding for face #{face_index + 1}")
        return emb.tolist()
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        raise CustomException(e,sys)

# ------------------ Function: Fetch Batch ------------------
def fetch_batch_from_supabase(offset, batch_size):
    try:
        logging.info(f"Fetching rows {offset} to {offset + batch_size - 1}")
        response = supabase.table(FTABLE_NAME)\
            .select("id, image_path, embedding")\
            .range(offset, offset + batch_size - 1)\
            .execute()
        return response.data
    except Exception as e:
        logging.error(f"Failed to fetch data from Supabase: {e}")
        raise CustomException(e,sys)

# ------------------ Function: Update Embedding ------------------
def update_embedding_in_supabase(record_id, embedding):
    try:
        supabase.table(FTABLE_NAME)\
            .update({"embedding": embedding})\
            .eq("id", record_id)\
            .execute()
        logging.info(f"Updated embedding for ID: {record_id}")
    except Exception as e:
        logging.error(f"Failed to update embedding for ID: {record_id} - {e}")
        raise CustomException(e,sys)

# ------------------ Function: Process One Row ------------------
def process_row(row):
    try:
        if row.get("embedding"):
            logging.info(f"[SKIP] ID {row['id']} already has embedding.")
            return False

        image_path = row.get("image_path")
        if not image_path:
            logging.warning(f"No image path for ID: {row['id']}")
            return False

        face_index = seen_images.get(image_path, 0)

        img = download_image(image_path)
        if img is None:
            return False

        embedding = get_embedding(img, face_index)
        if embedding is None:
            logging.warning(f"No face found in image: {image_path}")
            return False

        update_embedding_in_supabase(row["id"], embedding)

        seen_images[image_path] = face_index + 1
        return True

    except Exception as e:
        logging.error(f"Failed to process row ID {row.get('id')}: {e}")
        return False

# ------------------ Function: Run Main Pipeline ------------------
def run_embedding_pipeline(batch_size=1000):
    offset = 0
    total_processed = 0

    try:
        while True:
            data = fetch_batch_from_supabase(offset, batch_size)
            if not data:
                logging.info("[DONE] No more records to process.")
                break

            for row in data:
                if process_row(row):
                    total_processed += 1

            offset += batch_size
            time.sleep(1)

        logging.info(f"[DONE] Processed {total_processed} records.")
    except Exception as e:
        logging.critical(f"Pipeline execution failed: {e}")
        raise CustomException(e,sys)


# if __name__ == "__main__":
#     run_embedding_pipeline(batch_size=1000)