# 🎭 Actor Metadata Identifier API

A FastAPI-based web service that identifies actors in uploaded images using face recognition and embedding matching technology.

## 📋 Overview

This API provides endpoints to:
- **Identify actors** in uploaded images by matching face embeddings
- **Process face embeddings** through a batch pipeline
- **Handle multiple faces** in a single image with configurable similarity thresholds

## 🚀 Features

- **Face Detection & Recognition**: Automatically detects and identifies faces in images
- **Configurable Similarity Threshold**: Adjust matching sensitivity (default: 0.85)
- **Batch Processing Pipeline**: Efficiently process large datasets of face embeddings
- **Multi-face Support**: Handle images with multiple people
- **Performance Metrics**: Track execution time for each request
- **Comprehensive Error Handling**: Robust error management with detailed logging
- **CORS Support**: Cross-origin requests enabled for web applications
- **OpenAPI Documentation**: Auto-generated API documentation

## 🛠️ Technology Stack

- **FastAPI**: Modern Python web framework
- **PIL (Pillow)**: Image processing
- **Supabase**: Database for storing embeddings
- **Custom Face Recognition**: Proprietary face identification system

## 📦 Installation

### Prerequisites
- Python 3.8+
- Required dependencies (see requirements.txt)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd actor-metadata-identifier

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (if needed)
cp .env.example .env
# Edit .env with your configuration
```

## 🚦 Running the Application

### Development Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 📖 API Documentation

Once running, visit:
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## 🔌 API Endpoints

### 1. Identify Actor
**POST** `/identify-actor/`

Upload an image to identify actors using face recognition.

#### Request
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameters**:
  - `file` (required): Image file (JPEG, PNG, etc.)
  - `threshold` (optional): Similarity threshold (default: 0.85, range: 0.0-1.0)

#### Response
```json
{
  "faces_detected": 2,
  "results": [
    {
      "actor_name": "John Doe",
      "confidence": 0.92,
      "bounding_box": [x, y, width, height]
    },
    {
      "actor_name": "Jane Smith",
      "confidence": 0.88,
      "bounding_box": [x, y, width, height]
    }
  ],
  "execution_time_seconds": 1.23
}
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/identify-actor/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "threshold=0.85"
```

### 2. Run Embedding Pipeline
**POST** `/run-embedding-pipeline/`

Triggers the batch processing pipeline for face embeddings.

#### Request
- **Method**: POST
- **Parameters**:
  - `batch_size` (optional): Number of records to process per batch (default: 1000)

#### Response
```json
{
  "status": "success",
  "message": "Embedding pipeline completed successfully."
}
```

## 🏗️ Architecture

The project follows a modular architecture with clear separation of concerns:

### Core Components
- **`main.py`**: FastAPI application entry point with route definitions
- **`components/`**: Core business logic for face identification
- **`entity/`**: Data models and configuration entities
- **`utils/`**: Utility functions for database operations and model handling
- **`pipeline/`**: Data processing and transformation pipelines
- **`constants/`**: Application constants and configuration values
- **`exceptions/`**: Custom exception handling
- **`logger/`**: Centralized logging configuration

### Key Files
- **`face_identifier.py`**: Main face recognition and identification engine
- **`face_identifier_artifact_entity.py`**: Response data models
- **`face_identifier_config_entity.py`**: Configuration management
- **`con_embed_into_supabase.py`**: Database embedding operations
- **`models.py`**: Machine learning model utilities

### Threshold Settings
- **Low threshold (0.6-0.7)**: More permissive matching, higher chance of false positives
- **Medium threshold (0.8-0.85)**: Balanced accuracy and recall
- **High threshold (0.9-0.95)**: Strict matching, lower chance of false positives

### Environment Variables
Create a `.env` file with necessary configuration:
```env
# Database configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Logging level
LOG_LEVEL=INFO

# Other configurations...
```

## 📁 Project Structure

```
actor-metadata-identifier/
├── main.py                           # FastAPI application entry point
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   └── face_identifier.py        # Core face identification logic
│   ├── constants/
│   │   └── __init__.py               # Project constants and configurations
│   ├── entity/
│   │   ├── __init__.py
│   │   ├── face_identifier_artifact_entity.py  # Data models for face identification
│   │   └── face_identifier_config_entity.py    # Configuration entities
│   ├── exceptions/
│   │   └── __init__.py               # Custom exception classes
│   ├── logger/
│   │   └── __init__.py               # Logging utilities
│   ├── pipeline/
│   │   └── __init__.py               # Data processing pipelines
│   └── utils/
│       ├── __init__.py
│       ├── con_embed_into_supabase.py  # Supabase embedding utilities
│       └── models.py                   # ML model utilities
├── requirements.txt
└── README.md
```

## 🐛 Error Handling

The API handles various error scenarios:

- **400 Bad Request**: Invalid image format or parameters
- **500 Internal Server Error**: Processing errors or system issues

Common error responses:
```json
{
  "message": "Invalid image uploaded."
}
```

```json
{
  "message": "Internal server error",
  "details": "Specific error details"
}
```

## 📊 Performance

- **Response Time**: Typically 1-3 seconds per image
- **Supported Formats**: JPEG, PNG, WebP, and other PIL-supported formats
- **Concurrent Requests**: Supports multiple simultaneous requests
- **Batch Processing**: Configurable batch sizes for optimal performance

## 🧪 Testing

### Manual Testing
1. Start the server
2. Visit `http://localhost:8000/docs`
3. Use the interactive interface to test endpoints

### Example Test Images
- Single person portraits work best
- Group photos are supported
- Clear, well-lit images provide better accuracy

## 🚨 Troubleshooting

### Common Issues

1. **"Invalid image uploaded"**
   - Ensure the file is a valid image format
   - Check file size limits

2. **"Internal server error"**
   - Check server logs for detailed error information
   - Verify database connectivity

3. **Low accuracy results**
   - Try adjusting the threshold parameter
   - Ensure images are clear and faces are visible

### Logging
Logs are available in the console output and can be configured via the logging module.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 📞 Support

For issues, questions, or contributions:
- Create an issue in the repository
- Contact the development team
- Check the documentation at `/docs`

---

**Version**: 1.0  
**Last Updated**: [Current Date]