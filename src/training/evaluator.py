"""
Model Evaluation Pipeline.
Evaluates trained models on a fixed validation set.
"""
import logging
from pathlib import Path
from typing import Dict

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def evaluate_model(model_path: str, data_yaml: str, output_dir: str) -> Dict:
    """
    Evaluate a trained model and return metrics.
    """
    logger.info(f"Evaluating model {model_path}...")
    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        imgsz=640,
        batch=16,
        project=output_dir,
        name='evaluation',
        exist_ok=True,
        verbose=False,
        plots=True
    )

    results = {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr)
    }

    logger.info(f"Evaluation complete. mAP50: {results['mAP50']:.4f}")
    return results
