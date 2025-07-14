from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class FaceIdentifierArtifact:
    results: List[Dict[str, Any]]
    total_faces: int
