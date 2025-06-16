from PIL import Image
from io import BytesIO
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from supabase import create_client
import sys

from src.utils.models import mtcnn, resnet
from src.exceptions import CustomException
from src.logger import logging
from src.entity.face_identifier_config_entity import FaceIdentifierConfigEntity, ConfigEntity
from src.entity.face_identifier_artifact_entity import FaceIdentifierArtifact


class FaceIdentifier:
    def __init__(self):
        try:
            self.face_identifier_config = FaceIdentifierConfigEntity(config=ConfigEntity())

            self.supabase = create_client(
                self.face_identifier_config.supabase_url,
                self.face_identifier_config.supabase_key
            )

            logging.info("FaceIdentifier initialized successfully.")
        except Exception as e:
            raise CustomException(e,sys)

    def process_image(self, image_bytes: bytes, threshold: float = 0.85) -> FaceIdentifierArtifact:
        try:
            logging.info("Starting face identification process...")

            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            logging.info("Image loaded and converted to RGB.")

            face_tensors = mtcnn(image)
            if face_tensors is None:
                logging.warning("No faces detected in the uploaded image.")
                raise ValueError("No faces detected.")

            if isinstance(face_tensors, torch.Tensor):
                face_tensors = [face_tensors[i] for i in range(face_tensors.shape[0])]
            logging.info(f"{len(face_tensors)} face(s) detected.")

            logging.info("Fetching embeddings from Supabase.")
            response = self.supabase.table(self.face_identifier_config.table_name) \
                                     .select("name, dob, filmography, embedding") \
                                     .execute()
            raw_data = response.data

            valid_rows = [r for r in raw_data if isinstance(r["embedding"], list) and len(r["embedding"]) == 512]

            if not valid_rows:
                logging.error("No valid embeddings found in Supabase.")
                raise RuntimeError("No valid embeddings in Supabase.")

            db_embeddings = normalize(np.array([row["embedding"] for row in valid_rows]))
            logging.info(f"{len(valid_rows)} valid embeddings loaded from Supabase.")

            results = []

            for i, face_tensor in enumerate(face_tensors):
                face_tensor = face_tensor.unsqueeze(0).to(resnet.device)
                with torch.no_grad():
                    embedding = resnet(face_tensor).cpu().numpy()
                embedding = normalize(embedding)

                similarities = cosine_similarity(embedding, db_embeddings)[0]
                best_idx = int(np.argmax(similarities))
                best_score = similarities[best_idx]

                if best_score >= threshold:
                    matched = valid_rows[best_idx]
                    logging.info(f"Face {i} matched with {matched['name']} (score: {best_score:.3f})")
                    results.append({
                        "face_index": i,
                        "name": matched["name"],
                        "dob": matched["dob"],
                        "filmography": matched["filmography"],
                        "similarity_score": round(float(best_score), 3)
                    })
                else:
                    logging.info(f"Face {i} did not meet the threshold. Best score: {best_score:.3f}")
                    results.append({
                        "face_index": i,
                        "name": "Unknown",
                        "dob": "N/A",
                        "filmography": "N/A",
                        "similarity_score": round(float(best_score), 3)
                    })

            logging.info(f"Face identification completed. {len(results)} face(s) processed.")
            return FaceIdentifierArtifact(results=results, total_faces=len(face_tensors))

        except Exception as e:
            raise CustomException(e,sys)
