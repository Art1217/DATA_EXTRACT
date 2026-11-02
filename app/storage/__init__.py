from app.storage.base import Storage
from app.storage.local import LocalStorage

__all__ = ["Storage", "LocalStorage", "build_storage"]


def build_storage(settings) -> Storage:
    """El storage que pide `STORAGE_BACKEND`: disco local o bucket de GCS."""
    from app.config.settings import StorageBackend

    if settings.storage_backend is StorageBackend.GCS:
        from app.domain.errors import ConfigurationError
        from app.services import gcs_client
        from app.storage.gcs import GCSStorage

        if not settings.gcs_bucket:
            raise ConfigurationError("STORAGE_BACKEND=gcs requiere GCS_BUCKET.")
        bucket = gcs_client.client(settings.gcp_project_id, settings.credentials_path).bucket(
            settings.gcs_bucket
        )
        return GCSStorage(bucket, prefix=settings.gcs_prefix)
    return LocalStorage(settings.storage_dir)
