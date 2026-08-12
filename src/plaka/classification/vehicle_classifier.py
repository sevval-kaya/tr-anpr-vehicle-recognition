"""Vehicle make/model classification stage.

Wraps a timm backbone (EfficientNet/ConvNeXt/ViT — see docs/decisions.md for
the architecture chosen for the baseline). Preprocessing (resize/normalize)
is derived from the model's own timm data config rather than hardcoded, so
inference preprocessing always matches what the model was trained with.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from plaka.pipeline.schemas import MakeModelPrediction

if TYPE_CHECKING:
    import torch


class VehicleClassifier:
    """Fine-grained make/model classifier for a cropped vehicle image.

    `class_names_path` must list one label per line, in the same order as
    the model's output logits (the order used when the classifier head was
    trained).
    """

    def __init__(
        self,
        weights_path: str | Path,
        class_names_path: str | Path,
        architecture: str = "efficientnet_b0",
        device: str = "cpu",
    ) -> None:
        self._weights_path = Path(weights_path)
        self._class_names_path = Path(class_names_path)
        self._architecture = architecture
        self._device = device
        self._model: torch.nn.Module | None = None
        self._class_names: list[str] | None = None
        self._transform: Any = None

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import timm
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch and timm are required for VehicleClassifier; install with "
                "`pip install -e '.[classification]'`"
            ) from exc

        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"classifier weights not found at {self._weights_path}; "
                "train a checkpoint first (see scripts/, roadmap stage 2/4)"
            )
        if not self._class_names_path.exists():
            raise FileNotFoundError(f"class names file not found at {self._class_names_path}")

        self._class_names = self._class_names_path.read_text(encoding="utf-8").splitlines()

        model = timm.create_model(
            self._architecture, pretrained=False, num_classes=len(self._class_names)
        )
        state_dict = torch.load(self._weights_path, map_location=self._device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self._device)

        # timm.data doesn't ship a py.typed marker for these helpers.
        data_config = timm.data.resolve_data_config(  # type: ignore[attr-defined,no-untyped-call]
            {}, model=model
        )
        self._transform = timm.data.create_transform(  # type: ignore[attr-defined]
            **data_config, is_training=False
        )
        self._model = model

    def predict(self, vehicle_crop_bgr: NDArray[np.uint8], top_k: int = 5) -> MakeModelPrediction:
        """Classify a cropped vehicle image, returning the top-k make/model labels."""
        import cv2
        import torch
        import torch.nn.functional as functional
        from PIL import Image

        self._ensure_model_loaded()
        assert self._model is not None
        assert self._class_names is not None

        rgb = cv2.cvtColor(vehicle_crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(Image.fromarray(rgb)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probabilities = functional.softmax(logits, dim=1).squeeze(0)

        k = min(top_k, len(self._class_names))
        top_probs, top_indices = torch.topk(probabilities, k=k)

        return MakeModelPrediction(
            ranked_labels=[self._class_names[i] for i in top_indices.tolist()],
            ranked_confidences=top_probs.tolist(),
        )
