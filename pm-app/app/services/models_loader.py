import io
import logging
import os
from typing import Any
from urllib.parse import urlparse
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
    
    def _parse_uri(self, uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        return parsed.scheme, parsed.path
    
    def _load_from_local(self, path: str) -> Any:
        if path.startswith('/'):
            full_path = path
        else:
            full_path = os.path.join(self._config.model_storage.local_path, path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")
        
        return joblib.load(full_path)
    
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
        """
        Build model file path using convention {model_name}_{version}.joblib.
        Checks metadata for explicit path override.
        """
        if metadata and 'path' in metadata:
            return metadata['path']
        
        model_filename = f"{model_name}_{version}.joblib"
        
        if self._config.model_storage.backend == 'local':
            full_path = os.path.join(self._config.model_storage.local_path, model_filename)
            return f"file://{full_path}"
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
            
            # Eagerly load all attributes before session closes
            _ = artifact.id
            _ = artifact.model_name
            _ = artifact.version
            _ = artifact.model_metadata
            _ = artifact.promoted_at
            
            # Expunge from session so it can be used outside the context
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
        uri = f"file://{self._config.model_storage.local_path}/scaler.joblib"
        return self.load_model_from_uri(uri)
    
    def get_encoder(self) -> Any:
        uri = f"file://{self._config.model_storage.local_path}/encoder_engine_type.joblib"
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
