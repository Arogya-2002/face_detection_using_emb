from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from PIL import UnidentifiedImageError
import time

from src.components.face_identifier import FaceIdentifier
from src.entity.face_identifier_artifact_entity import FaceIdentifierArtifact
from src.exceptions import CustomException
from src.logger import logging
from src.utils.con_embed_into_supabase import run_embedding_pipeline

face_identifier = FaceIdentifier()

app = FastAPI(title="🎭 Actor Metadata Identifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/identify-actor/")
async def identify_actor(file: UploadFile = File(...), threshold: float = Form(0.85)):
    start = time.time()
    try:
        image_bytes = await file.read()
        logging.info(f"Received image for actor identification. Threshold: {threshold}")

        artifact: FaceIdentifierArtifact = face_identifier.process_image(image_bytes, threshold)

        logging.info(f"Actor identification completed in {round(time.time() - start, 2)} seconds.")

        return {
            "faces_detected": artifact.total_faces,
            "results": artifact.results,
            "execution_time_seconds": round(time.time() - start, 2)
        }

    except UnidentifiedImageError:
        logging.warning("Unidentified image format.")
        return JSONResponse(status_code=400, content={"message": "Invalid image uploaded."})
    except ValueError as ve:
        logging.warning(f"ValueError during identification: {str(ve)}")
        return JSONResponse(status_code=400, content={"message": str(ve)})
    except RuntimeError as re:
        logging.error(f"RuntimeError during identification: {str(re)}")
        return JSONResponse(status_code=500, content={"message": str(re)})
    except Exception as e:
        logging.exception("Unhandled exception occurred.")
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error", "details": str(e)}
        )


@app.get("/openapi.json", include_in_schema=False)
async def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version="1.0",
        routes=app.routes,
        description="Upload an image to identify actors by matching face embeddings."
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


@app.post("/run-embedding-pipeline/")
async def trigger_embedding_pipeline(batch_size: int = 1000):
    """
    Triggers the embedding pipeline to process and store face embeddings.
    """
    try:
        logging.info(f"Pipeline triggered via API with batch_size={batch_size}")
        run_embedding_pipeline(batch_size=batch_size)
        return {"status": "success", "message": "Embedding pipeline completed successfully."}
    except Exception as e:
        logging.exception("Embedding pipeline execution failed.")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Pipeline execution failed", "details": str(e)}
        )