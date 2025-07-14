from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from PIL import UnidentifiedImageError
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from src.components.face_identifier import FaceIdentifier
from src.entity.face_identifier_artifact_entity import FaceIdentifierArtifact
from src.exceptions import CustomException
from src.logger import logging
from src.utils.con_embed_into_supabase import run_embedding_pipeline
from src.pipeline.face_detection_pipeline import run_face_detection_pipeline


app = FastAPI(title="🎭 Actor Metadata Identifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/detect-faces/")
async def detect_faces(
    file: UploadFile = File(...),
    threshold: float = Query(0.85, ge=0.0, le=1.0)
):
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG or PNG.")

        image_bytes = await file.read()
        result = run_face_detection_pipeline(image_bytes, threshold)
        return JSONResponse(content=result)

    except Exception as e:
        logging.exception("Failed during face detection API call")
        raise HTTPException(status_code=500, detail=str(e))

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
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)