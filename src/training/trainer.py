"""
YOLO Training Pipeline.
Trains one model per call, starting from pretrained weights with a fixed seed.
"""
import logging
from pathlib import Path
from typing import Dict, Optional

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def device_kwargs(config: Dict) -> Dict:
    """Only pass `device` when the config names one; 'auto' lets Ultralytics pick."""
    device = config.get('device', 'auto')
    return {} if device in (None, '', 'auto') else {'device': device}


def train_model(data_yaml: str, output_dir: str, config: Dict, seed: Optional[int] = None) -> Path:
    """
    Fine-tune a pretrained YOLO model on `data_yaml` and return the path to best.pt.

    A fresh model is loaded on every call so nothing leaks between experiments. `seed` overrides
    config['seed'] so the same variant can be trained several times.
    """
    output_path = Path(output_dir).resolve()   # absolute: a relative project gets a runs/detect/ prefix
    output_path.mkdir(parents=True, exist_ok=True)

    seed = config.get('seed', 42) if seed is None else seed
    logger.info(f"Training model on {data_yaml} (seed {seed})...")

    model = YOLO(config.get('model_name', 'yolov8n.pt'))
    results = model.train(
        data=data_yaml,
        epochs=config.get('epochs', 30),
        imgsz=config.get('img_size', 640),
        batch=config.get('batch_size', 16),
        seed=seed,
        deterministic=True,
        optimizer=config.get('optimizer', 'auto'),
        project=str(output_path),
        name='experiment',
        exist_ok=True,
        verbose=False,
        plots=True,
        **device_kwargs(config),
    )

    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    logger.info(f"Training complete. Best model saved to {best_model_path}")
    return best_model_path
