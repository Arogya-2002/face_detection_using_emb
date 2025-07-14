import time
from src.exceptions import CustomException
from src.logger import logging
from src.components.face_identifier import FaceIdentifier
import sys
def run_face_detection_pipeline(image_bytes: bytes, threshold: float = 0.85) -> dict:
    try:
        logging.info(f"Starting face detection pipeline with threshold: {threshold}")
        start_time = time.time()

        face_identifier = FaceIdentifier()
        artifact = face_identifier.process_image(image_bytes, threshold)

        end_time = time.time()
        execution_time = round(end_time - start_time, 3)

        logging.info("Face detection pipeline completed successfully.")
        return {
            "faces_detected": artifact.total_faces,
            "results": artifact.results,
            "execution_time_seconds": execution_time
        }

    except Exception as e:
        logging.error(f"Face detection pipeline failed: {e}")
        raise CustomException(e, sys)

# if __name__ == "__main__":
#     # Example usage
#     with open(r"/Users/vamshi/Desktop/projects/Litzchill/face_detection/mahexsh.jpeg", "rb") as img_file:
#         image_bytes = img_file.read()
#     result = run_face_detection_pipeline(image_bytes, threshold=0.85)
#     print(result)