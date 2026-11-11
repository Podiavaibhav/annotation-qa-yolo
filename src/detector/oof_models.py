"""
Out-of-fold models for the detector.

The images are split into k folds. For each fold one model is trained on the OTHER folds, and
that model is used to predict only the held-out fold. So no image is ever predicted by a model
that trained on it, which keeps the model from simply repeating the labels it memorised.
"""
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import yaml
from ultralytics import YOLO

from src.dataset_loader import YOLODatasetLoader
from src.training.trainer import device_kwargs

logger = logging.getLogger(__name__)


def make_folds(names: List[str], k: int, seed: int) -> List[List[str]]:
    """Shuffle deterministically and deal the names into k folds of near-equal size."""
    if k < 2:
        raise ValueError("Out-of-fold prediction needs at least 2 folds.")
    names = sorted(names)
    if len(names) < k:
        raise ValueError(f"Need at least {k} images for {k} folds, got {len(names)}.")
    random.Random(seed).shuffle(names)
    return [names[i::k] for i in range(k)]


def train_oof_models(dataset_path: str, config: Dict, k: int = None, seed: int = None,
                     output_dir: str = None) -> Dict[str, str]:
    """
    Train one model per fold on the other folds' labels.

    Returns a dict mapping each training image's file name to the weights of the model that did
    NOT see it, ready to pass to ModelDisagreementDetector(fold_models=...).
    """
    k = k or config.get('oof_folds', 2)
    seed = config.get('seed', 42) if seed is None else seed
    out = Path(output_dir or Path(config.get('output_dir', 'outputs')) / 'oof').resolve()
    out.mkdir(parents=True, exist_ok=True)

    loader = YOLODatasetLoader(dataset_path)
    pairs = loader.get_image_label_pairs('train')
    image_by_name = {img.name: img for img, _ in pairs}
    folds = make_folds(list(image_by_name), k, seed)

    try:
        val_images = loader.get_split_paths('val')[0]
    except FileNotFoundError:
        val_images = loader.get_split_paths('train')[0]   # Ultralytics insists on a val set

    mapping: Dict[str, str] = {}
    for i, held_out in enumerate(folds):
        train_names = [n for j, fold in enumerate(folds) if j != i for n in fold]
        list_file = out / f'fold{i}_train.txt'
        list_file.write_text("\n".join(str(image_by_name[n].resolve()) for n in train_names) + "\n")

        yaml_path = out / f'fold{i}.yaml'
        with open(yaml_path, 'w') as f:
            yaml.safe_dump({'path': str(out), 'train': str(list_file), 'val': str(val_images.resolve()),
                            'names': loader.config.get('names')}, f, sort_keys=False)

        weights = out / 'runs' / f'fold{i}' / 'weights' / 'last.pt'
        if weights.exists() and config.get('resume', False):   # only when asked: data may have changed
            logger.info(f"Fold {i}: reusing {weights}")
        else:
            logger.info(f"Fold {i}: training on {len(train_names)} images, predicting {len(held_out)}")
            model = YOLO(config.get('model_name', 'yolov8n.pt'))
            results = model.train(
                data=str(yaml_path),
                epochs=config.get('oof_epochs', config.get('epochs', 30)),
                imgsz=config.get('img_size', 640),
                batch=config.get('batch_size', 16),
                seed=seed,
                project=str(out / 'runs'),
                name=f'fold{i}',
                exist_ok=True,
                val=False,
                plots=False,
                verbose=False,
                **device_kwargs(config),
            )
            save_dir = Path(getattr(results, 'save_dir', None) or out / 'runs' / f'fold{i}')
            weights = save_dir / 'weights' / 'last.pt'
            if not weights.exists():
                weights = save_dir / 'weights' / 'best.pt'

        for name in held_out:
            mapping[name] = str(weights)

    (out / 'oof_models.json').write_text(json.dumps(mapping, indent=2))
    return mapping
