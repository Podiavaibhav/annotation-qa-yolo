"""
YOLO Training Pipeline.
Trains models using strict reproducibility constraints.
"""
import logging
from pathlib import Path
from typing import Dict

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def train_model(data_yaml: str, output_dir: str, config: Dict) -> Path:
    """
    Train a YOLOv8 model from scratch.
    Always loads fresh pretrained weights to prevent leakage between experiments.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training model on {data_yaml}...")
    
    # Load fresh weights
    model = YOLO(config.get('model_name', 'yolov8n.pt'))

    results = model.train(
        data=data_yaml,
        epochs=config.get('epochs', 30),
        imgsz=config.get('img_size', 640),
        batch=config.get('batch_size', 16),
        seed=config.get('seed', 42),
        optimizer=config.get('optimizer', 'auto'),
        project=str(output_path),
        name='experiment',
        exist_ok=True,
        verbose=False,
        plots=True
    )

    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    logger.info(f"Training complete. Best model saved to {best_model_path}")
    return best_model_path
