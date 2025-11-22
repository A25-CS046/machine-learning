import io
import logging
import os
import pathlib
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname
import joblib
from sqlalchemy import desc
from app.config import load_config
from app.db import get_db
from app.models import ModelArtifact

logger = logging.getLogger(__name__)


class ModelsLoader:
    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._config = load_config()
        # Calculate absolute project root (pm-app)
        # __file__ = app/services/models_loader.py -> parent=services -> parent=app -> parent=pm-app
        self._base_dir = pathlib.Path(__file__).resolve().parent.parent.parent

    def _parse_uri(self, uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        return parsed.scheme, parsed.path
    
    def _load_from_local(self, path_from_uri: str) -> Any:
        """
        Robust local loader that handles Windows paths and fallbacks.
        """
        # 1. Convert URI path (e.g., /C:/Users/...) to system path (C:\Users\...)
        clean_path = url2pathname(path_from_uri)
        
        # 2. Primary attempt: Use the resolved path directly
        if os.path.exists(clean_path):
            return joblib.load(clean_path)

        # 3. Fallback Strategy: If exact path fails, look in common locations
        filename = os.path.basename(clean_path)
        
        # Define search candidates. 
        # self._base_dir = '.../pm-app'
        # self._base_dir.parent = '.../' (The folder containing pm-app)
        candidates = [
            clean_path,                                      # 1. Original attempt
            self._base_dir.parent / 'artifacts' / filename,  # 2. Sibling: ../artifacts/file (OUTSIDE pm-app)
            self._base_dir / 'artifacts' / filename,         # 3. Child: pm-app/artifacts/file
            self._base_dir / filename,                       # 4. Root: pm-app/file
            self._base_dir / 'app' / 'artifacts' / filename  # 5. Nested: pm-app/app/artifacts/file
        ]

        for candidate in candidates:
            # Convert pathlib objects to string for os.path.exists
            candidate_str = str(candidate)
            if os.path.exists(candidate_str):
                logger.info(f"Found model file via fallback at: {candidate_str}")
                return joblib.load(candidate_str)

        # 4. Debugging: If ALL fail, report on the sibling directory specifically
        # since that is the user's expected location.
        sibling_artifacts = self._base_dir.parent / 'artifacts'
        if os.path.exists(sibling_artifacts):
            files = os.listdir(sibling_artifacts)
            logger.error(f"File '{filename}' not found in sibling artifacts folder ({sibling_artifacts}). Existing files: {files}")
        else:
            logger.error(f"Sibling artifacts directory not found: {sibling_artifacts}")
            
        raise FileNotFoundError(f"Model file not found: {clean_path} (Checked fallbacks including ../artifacts)")
    
    def _load_from_gcs(self, path: str) -> Any:
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
        
        if not self._config.model_storage.gcs_bucket:
            raise ValueError("GCS_MODEL_BUCKET not configured")
        
        parts = path.lstrip('/').split('/', 1)
        if len(parts) == 2:
            bucket_name, blob_path = parts
        else:
            bucket_name = self._config.model_storage.gcs_bucket
            blob_path = parts[0]
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        blob_bytes = blob.download_as_bytes()
        return joblib.load(io.BytesIO(blob_bytes))
    
    def load_model_from_uri(self, uri: str) -> Any:
        if uri in self._cache:
            logger.debug(f"Model cache hit: {uri}")
            return self._cache[uri]
        
        scheme, path = self._parse_uri(uri)
        
        logger.info(f"Loading model from URI: {uri}")
        
        if scheme == 'file' or not scheme:
            model = self._load_from_local(path)
        elif scheme == 'gs':
            model = self._load_from_gcs(path)
        else:
            raise ValueError(f"Unsupported model URI scheme: {scheme}")
        
        self._cache[uri] = model
        logger.info(f"Model loaded and cached: {uri}")
        return model
    
    def _build_model_path(self, model_name: str, version: str, metadata: dict | None = None) -> str:
        if metadata and 'path' in metadata:
            return metadata['path']
        
        model_filename = f"{model_name}_{version}.joblib"
        
        if self._config.model_storage.backend == 'local':
            # Construct absolute path based on config
            full_path = os.path.join(self._config.model_storage.local_path, model_filename)
            # Convert to valid file URI (file:///C:/...) using pathlib
            return pathlib.Path(os.path.abspath(full_path)).as_uri()
        elif self._config.model_storage.backend == 'gcs':
            return f"gs://{self._config.model_storage.gcs_bucket}/{model_filename}"
        else:
            raise ValueError(f"Unknown storage backend: {self._config.model_storage.backend}")
    
    def get_latest_model_artifact(self, model_name: str) -> ModelArtifact:
        with get_db() as session:
            artifact = session.query(ModelArtifact).filter(
                ModelArtifact.model_name == model_name
            ).order_by(desc(ModelArtifact.promoted_at)).first()
            
            if not artifact:
                raise ValueError(f"No model artifact found for: {model_name}")
            
            _ = artifact.id
            _ = artifact.model_name
            _ = artifact.version
            _ = artifact.model_metadata
            _ = artifact.promoted_at
            
            session.expunge(artifact)
            return artifact
    
    def get_model(self, model_name: str) -> tuple[Any, ModelArtifact]:
        artifact = self.get_latest_model_artifact(model_name)
        model_path = self._build_model_path(
            artifact.model_name,
            artifact.version,
            artifact.model_metadata
        )
        model = self.load_model_from_uri(model_path)
        return model, artifact
    
    def get_classification_model(self, model_name: str = 'xgb_classifier') -> tuple[Any, ModelArtifact]:
        return self.get_model(model_name)
    
    def get_forecast_model(self, model_name: str = 'xgb_regressor') -> tuple[Any, ModelArtifact]:
        return self.get_model(model_name)
    
    def get_scaler(self) -> Any:
        # Build absolute path to scaler and convert to proper file:// URI
        full_path = os.path.join(self._config.model_storage.local_path, "scaler.joblib")
        uri = pathlib.Path(os.path.abspath(full_path)).as_uri()
        return self.load_model_from_uri(uri)
    
    def get_encoder(self) -> Any:
        # Build absolute path to encoder and convert to proper file:// URI
        full_path = os.path.join(self._config.model_storage.local_path, "encoder_engine_type.joblib")
        uri = pathlib.Path(os.path.abspath(full_path)).as_uri()
        return self.load_model_from_uri(uri)
    
    def reload_all(self) -> None:
        logger.info("Clearing model cache")
        self._cache.clear()
        logger.info("Model cache cleared")
    
    def get_cached_models(self) -> list[str]:
        return list(self._cache.keys())


_loader_instance: ModelsLoader | None = None


def get_models_loader() -> ModelsLoader:
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ModelsLoader()
    return _loader_instance