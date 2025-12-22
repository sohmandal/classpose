"""
Specifies model configuration data models, associated utilities and default
model configurations.
"""

import os
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

from classpose.log import get_logger
from classpose.utils import download_if_unavailable

logger = get_logger("classpose.model_configs")

HOME = Path.home()

ROOT_MODEL_DIR = os.getenv("CLASSPOSE_MODEL_DIR", HOME / ".classpose_models")
ROOT_MODEL_DIR = Path(ROOT_MODEL_DIR)
REPO_ID = "classpose/classpose"
DEFAULT_MODEL_CONFIGS = {
    "conic": {
        "path": str(ROOT_MODEL_DIR / "conic.pt"),
        "mpp": 0.5,
        "url": None,
        "hf": {"repo_id": REPO_ID, "filename": "conic.pt"},
        "cell_types": [
            "Neutrophil",
            "Epithelial",
            "Lymphocyte",
            "Plasma cell",
            "Eosinophil",
            "Connective",
        ],
    },
    "consep": {
        "path": str(ROOT_MODEL_DIR / "consep.pt"),
        "mpp": 0.25,
        "url": None,
        "hf": {"repo_id": REPO_ID, "filename": "consep.pt"},
        "cell_types": [
            "Other",
            "Inflammatory",
            "Healthy epithelial",
            "Malignant epithelial",
            "Stroma",
            "Muscle",
        ],
    },
    "glysac": {
        # MPP is presumed, collected using 40x from "Aperio digital scanner"
        "path": str(ROOT_MODEL_DIR / "glysac.pt"),
        "mpp": 0.25,
        "url": None,
        "hf": {"repo_id": REPO_ID, "filename": "glysac.pt"},
        "cell_types": [
            "Other",
            "Lymphocyte",
            "Epithelial",
            "Ambiguous",
        ],
    },
    "monusac": {
        # MPP is presumed, collected from multiple TCGA slides
        "path": str(ROOT_MODEL_DIR / "monusac.pt"),
        "mpp": 0.25,
        "url": None,
        "hf": {"repo_id": REPO_ID, "filename": "monusac.pt"},
        "cell_types": [
            "Epithelial",
            "Lymphocyte",
            "Macrophage",
            "Neutrophil",
        ],
    },
    "nucls": {
        "path": str(ROOT_MODEL_DIR / "nucls.pt"),
        "mpp": 0.2,
        "url": None,
        "hf": {"repo_id": REPO_ID, "filename": "nucls.pt"},
        "cell_types": [
            "Tumor",
            "Stroma",
            "Lymphocyte",
            "Plasma cell",
            "Macrophage",
            "Other",
        ],
    },
    "puma": {
        "path": str(ROOT_MODEL_DIR / "puma.pt"),
        "mpp": 0.22,
        "url": None,
        "hf": {"repo_id": REPO_ID, "filename": "puma.pt"},
        "cell_types": [
            "Apoptosis",
            "Tumor",
            "Endothelial",
            "Stroma",
            "Lymphocyte",
            "Histocyte",
            "Epithelial",
            "Melanophage",
            "Other",
        ],
    },
}


class HuggingFaceConfig(BaseModel):
    """
    HuggingFace model configuration model.
    """

    repo_id: str
    filename: str


class ModelConfig(BaseModel):
    """
    Classpose model configuration model.
    """

    path: str
    mpp: float
    url: str | None = None
    hf: HuggingFaceConfig | None = None
    cell_types: list[str]

    @staticmethod
    def load_from_yaml(path: str) -> "ModelConfig":
        """
        Loads a model configuration from a YAML file.

        Args:
            path (str): Path to the YAML file.

        Returns:
            ModelConfig: The loaded model configuration.
        """
        logger.info(f"Loading model config from {path}")
        with open(path) as o:
            config = yaml.safe_load(o)
        if "hf" in config:
            config["hf"] = HuggingFaceConfig(**config["hf"])
        return ModelConfig(**config)

    def download_if_necessary(self) -> None:
        """
        Downloads the model weights if they are not available locally.
        """
        if Path(self.path).exists():
            logger.info("Model weights already in %s", self.path)
            return
        logger.info("Downloading model weights to %s", self.path)
        if self.url is not None:
            logger.info("Downloading model weights from %s", self.url)
            download_if_unavailable(self.path, self.url)
        elif self.hf is not None:
            logger.info("Downloading model weights from Hugging Face")
            hf_token = os.getenv("HF_TOKEN", None)
            local_dir = str(Path(self.path).parent)
            if hf_token is None:
                hf_hub_download(
                    repo_id=self.hf.repo_id,
                    filename=self.hf.filename,
                    local_dir=local_dir,
                )
            else:
                hf_hub_download(
                    repo_id=self.hf.repo_id,
                    filename=self.hf.filename,
                    token=hf_token,
                    local_dir=local_dir,
                )
