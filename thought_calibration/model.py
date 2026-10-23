"""Serializable StandardScaler → PCA → logistic correctness probe."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn


class ThoughtCalibrationModel:
    MODEL_VERSION = 1

    def __init__(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        scaler_options: dict[str, Any] | None = None,
        pca_options: dict[str, Any] | None = None,
        probe_options: dict[str, Any] | None = None,
    ) -> None:
        self.metadata = {"model_version": self.MODEL_VERSION, **(metadata or {})}
        self.scaler = StandardScaler(**(scaler_options or {}))
        self.pca = PCA(**(pca_options or {}))
        self.probe = LogisticRegression(**(probe_options or {}))

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "ThoughtCalibrationModel":
        features = np.asarray(features)
        labels = np.asarray(labels)
        if features.ndim != 2:
            raise ValueError("features must have shape [N, hidden_size]")
        if len(np.unique(labels)) != 2:
            raise ValueError("Training labels must contain both classes")
        n_components = self.pca.n_components
        if isinstance(n_components, int) and n_components > min(features.shape):
            raise ValueError(
                f"PCA n_components={n_components} exceeds min(training samples, hidden size)="
                f"{min(features.shape)}"
            )
        scaled = self.scaler.fit_transform(features)
        reduced = self.pca.fit_transform(scaled)
        self.probe.fit(reduced, labels)
        self.metadata.setdefault("hidden_size", int(features.shape[1]))
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features)
        if features.ndim != 2:
            raise ValueError("features must have shape [N, hidden_size]")
        hidden_size = self.metadata.get("hidden_size")
        if hidden_size is not None and features.shape[1] != hidden_size:
            raise ValueError(f"Expected hidden size {hidden_size}, got {features.shape[1]}")
        reduced = self.pca.transform(self.scaler.transform(features))
        positive = int(np.flatnonzero(self.probe.classes_ == 1)[0])
        return self.probe.predict_proba(reduced)[:, positive]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "pca": self.pca,
                "probe": self.probe,
                "metadata": self.metadata,
            },
            path,
        )

    def to_torch(
        self, *, dtype: Optional[torch.dtype] = None
    ) -> "ThoughtCalibrationModule":
        """Create a frozen tensor-only module with the same probabilities."""
        return ThoughtCalibrationModule.from_model(self, dtype=dtype)

    @classmethod
    def load(cls, path: str | Path) -> "ThoughtCalibrationModel":
        payload = joblib.load(path)
        model = cls(metadata=payload["metadata"])
        model.scaler = payload["scaler"]
        model.pca = payload["pca"]
        model.probe = payload["probe"]
        if model.metadata.get("model_version") != cls.MODEL_VERSION:
            raise ValueError(f"Unsupported model version: {model.metadata.get('model_version')}")
        return model


class ThoughtCalibrationModule(nn.Module):
    """Tensor-only inference wrapper for a fitted Thought Calibration probe.

    The last input dimension must be the trained hidden size. All preceding
    dimensions are treated as independent pooled features and are preserved in
    the probability output. Callers remain responsible for segment pooling,
    matching the training metadata's ``segment_mean`` policy.
    """

    def __init__(
        self,
        *,
        scaler_mean: torch.Tensor,
        scaler_scale: torch.Tensor,
        pca_mean: torch.Tensor,
        pca_components: torch.Tensor,
        pca_scale: torch.Tensor,
        probe_weight: torch.Tensor,
        probe_bias: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)
        self.register_buffer("pca_mean", pca_mean)
        self.register_buffer("pca_components", pca_components)
        self.register_buffer("pca_scale", pca_scale)
        self.register_buffer("probe_weight", probe_weight)
        self.register_buffer("probe_bias", probe_bias)
        self.metadata = dict(metadata or {})

    @classmethod
    def from_model(
        cls,
        model: ThoughtCalibrationModel,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> "ThoughtCalibrationModule":
        if not hasattr(model.probe, "classes_"):
            raise ValueError("ThoughtCalibrationModel must be fitted before conversion")
        classes = np.asarray(model.probe.classes_)
        if classes.tolist() != [0, 1] or model.probe.coef_.shape[0] != 1:
            raise ValueError("Torch wrapper requires a fitted binary probe with classes [0, 1]")
        hidden_size = int(model.scaler.n_features_in_)
        scaler_mean = model.scaler.mean_ if model.scaler.with_mean else np.zeros(hidden_size)
        scaler_scale = model.scaler.scale_ if model.scaler.with_std else np.ones(hidden_size)
        pca_scale = (
            np.sqrt(model.pca.explained_variance_)
            if model.pca.whiten
            else np.ones(model.pca.n_components_)
        )
        values_to_store = (
            scaler_mean,
            scaler_scale,
            model.pca.mean_,
            model.pca.components_,
            pca_scale,
            model.probe.coef_[0],
            model.probe.intercept_[0],
        )
        if dtype is None:
            # sklearn may store scaler statistics and fitted estimators with
            # different NumPy dtypes. Use their lossless common dtype so all
            # torch matmuls are compatible without a float32 coercion.
            numpy_dtype = np.result_type(*(np.asarray(value).dtype for value in values_to_store))
            dtype = torch.from_numpy(np.empty((), dtype=numpy_dtype)).dtype

        def tensor(value: Any) -> torch.Tensor:
            result = torch.as_tensor(np.asarray(value))
            return result.to(dtype=dtype).clone()

        return cls(
            scaler_mean=tensor(scaler_mean),
            scaler_scale=tensor(scaler_scale),
            pca_mean=tensor(model.pca.mean_),
            pca_components=tensor(model.pca.components_),
            pca_scale=tensor(pca_scale),
            probe_weight=tensor(model.probe.coef_[0]),
            probe_bias=tensor(model.probe.intercept_[0]),
            metadata=model.metadata,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> "ThoughtCalibrationModule":
        """Load ``model.joblib`` directly as a PyTorch module."""
        path = Path(path)
        if path.is_dir():
            path = path / "model.joblib"
        return ThoughtCalibrationModel.load(path).to_torch(dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        new_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        del attention_mask, position_ids, new_tokens
        if not torch.is_tensor(hidden_states):
            raise TypeError("hidden_states must be a torch.Tensor")
        if hidden_states.ndim < 1 or hidden_states.shape[-1] != self.scaler_mean.numel():
            raise ValueError(
                f"Expected hidden_states shape [..., {self.scaler_mean.numel()}], "
                f"got {tuple(hidden_states.shape)}"
            )
        values = hidden_states.to(dtype=self.scaler_mean.dtype)
        scaled = (values - self.scaler_mean) / self.scaler_scale
        reduced = torch.matmul(scaled - self.pca_mean, self.pca_components.transpose(0, 1))
        reduced = reduced / self.pca_scale.clamp_min(torch.finfo(reduced.dtype).eps)
        logits = torch.matmul(reduced, self.probe_weight) + self.probe_bias
        return torch.sigmoid(logits)
