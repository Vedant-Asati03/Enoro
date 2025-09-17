"""
Model management utilities for ML pipelines.
"""

import pickle
import joblib
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class ModelManager:
    """
    Utilities for saving, loading, and managing ML models.

    Provides consistent model persistence and versioning
    across different ML components.
    """

    def __init__(self, models_dir: str = "/tmp/enoro_models"):
        """
        Initialize model manager.

        Args:
            models_dir: Directory to store model files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Model registry file
        self.registry_file = self.models_dir / "model_registry.json"
        self._load_registry()

    def _load_registry(self):
        """Load model registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def _save_registry(self):
        """Save model registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def save_model(
        self,
        model: Any,
        model_name: str,
        version: str = "latest",
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Save a model to disk with metadata.

        Args:
            model: Model object to save
            model_name: Name identifier for the model
            version: Version string (defaults to "latest")
            metadata: Optional metadata dict

        Returns:
            Path to saved model file
        """
        # Create model filename
        filename = f"{model_name}_v{version}.joblib"
        model_path = self.models_dir / filename

        # Save model using joblib (works for sklearn models)
        try:
            joblib.dump(model, model_path)
        except Exception:
            # Fallback to pickle for other objects
            with open(model_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(model, f)
            model_path = model_path.with_suffix(".pkl")

        # Update registry
        self.registry[model_name] = {
            "version": version,
            "path": str(model_path),
            "saved_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self._save_registry()

        return str(model_path)

    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """
        Load a model from disk.

        Args:
            model_name: Name of the model to load
            version: Version to load (defaults to "latest")

        Returns:
            Loaded model object

        Raises:
            FileNotFoundError: If model doesn't exist
        """
        if model_name not in self.registry:
            raise FileNotFoundError(f"Model '{model_name}' not found in registry")

        model_info = self.registry[model_name]
        model_path = Path(model_info["path"])

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load based on file extension
        if model_path.suffix == ".joblib":
            return joblib.load(model_path)
        elif model_path.suffix == ".pkl":
            with open(model_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported model file format: {model_path.suffix}")

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get metadata about a saved model.

        Args:
            model_name: Name of the model

        Returns:
            Model info dict or None if not found
        """
        return self.registry.get(model_name)

    def list_models(self) -> Dict[str, Dict]:
        """
        List all saved models and their info.

        Returns:
            Dictionary of model_name -> model_info
        """
        return self.registry.copy()

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from disk and registry.

        Args:
            model_name: Name of the model to delete

        Returns:
            True if successfully deleted, False if not found
        """
        if model_name not in self.registry:
            return False

        model_info = self.registry[model_name]
        model_path = Path(model_info["path"])

        # Delete file if it exists
        if model_path.exists():
            model_path.unlink()

        # Remove from registry
        del self.registry[model_name]
        self._save_registry()

        return True

    def cleanup_old_versions(self, keep_latest: int = 3):
        """
        Clean up old model versions, keeping only the latest N.

        Args:
            keep_latest: Number of latest versions to keep per model
        """
        # Group models by base name
        model_groups = {}
        for model_name, info in self.registry.items():
            base_name = model_name.split("_v")[0] if "_v" in model_name else model_name
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append((model_name, info))

        # Sort by saved_at date and keep only latest versions
        for base_name, models in model_groups.items():
            if len(models) > keep_latest:
                # Sort by saved_at (newest first)
                models.sort(key=lambda x: x[1]["saved_at"], reverse=True)

                # Delete older versions
                for model_name, info in models[keep_latest:]:
                    self.delete_model(model_name)


class ModelPerformanceTracker:
    """Track model performance metrics over time."""

    def __init__(self, models_dir: str = "/tmp/enoro_models"):
        """Initialize performance tracker."""
        self.models_dir = Path(models_dir)
        self.metrics_file = self.models_dir / "performance_metrics.json"
        self._load_metrics()

    def _load_metrics(self):
        """Load performance metrics from disk."""
        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {}

    def _save_metrics(self):
        """Save performance metrics to disk."""
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def record_performance(
        self,
        model_name: str,
        metrics: Dict[str, float],
        dataset_info: Optional[Dict] = None,
    ):
        """
        Record performance metrics for a model.

        Args:
            model_name: Name of the model
            metrics: Performance metrics dict
            dataset_info: Optional info about test dataset
        """
        if model_name not in self.metrics:
            self.metrics[model_name] = []

        record = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "dataset_info": dataset_info or {},
        }

        self.metrics[model_name].append(record)
        self._save_metrics()

    def get_performance_history(self, model_name: str) -> list:
        """
        Get performance history for a model.

        Args:
            model_name: Name of the model

        Returns:
            List of performance records
        """
        return self.metrics.get(model_name, [])
