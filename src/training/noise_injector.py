"""
Synthetic Noise Injection Utility.
Corrupts training labels to simulate annotation errors.

Every corruption is written to corruption_manifest.json, so the detector can later be scored
against what was really broken (see src/detector/scoring.py).
"""
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from src.dataset_loader import YOLODatasetLoader
from src.yaml_utils import set_yaml_root

logger = logging.getLogger(__name__)

DEFAULTS = {
    'delete_probability': 0.08,
    'class_flip_probability': 0.08,
    'coordinate_shift_probability': 0.08,
    'resize_probability': 0.06,
    'duplicate_probability': 0.04,
    'shift_min': 0.2,      # a shift moves the box by shift_min..shift_amount of its own size
    'shift_amount': 0.5,
    'resize_min': 0.25,    # a resize scales width and height by 1 +/- (resize_min..resize_amount)
    'resize_amount': 0.5,
}


def _iou(a: Dict, b: Dict) -> float:
    ax1, ay1, ax2, ay2 = a['x'] - a['w'] / 2, a['y'] - a['h'] / 2, a['x'] + a['w'] / 2, a['y'] + a['h'] / 2
    bx1, by1, bx2, by2 = b['x'] - b['w'] / 2, b['y'] - b['h'] / 2, b['x'] + b['w'] / 2, b['y'] + b['h'] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a['w'] * a['h'] + b['w'] * b['h'] - inter
    return inter / union if union > 0 else 0.0


def _clip_center(c: float, size: float) -> float:
    """Keep a box of this size fully inside the image."""
    lo, hi = size / 2, 1 - size / 2
    return 0.5 if lo > hi else float(min(max(c, lo), hi))


class NoiseInjector:
    """Injects controlled noise into a YOLO dataset for robustness testing."""

    def __init__(self, dataset_path: str, config: Dict):
        path = Path(dataset_path)
        self.dataset_path = path.parent if path.is_file() else path
        self.config = config
        self.noise_config = {**DEFAULTS, **config.get('noise', {})}
        self.output_dir = Path(config.get('output_dir', 'outputs')) / 'noisy_dataset'
        self.manifest: List[Dict] = []

        # Dedicated generators so the global random state is untouched
        seed = config.get('seed', 42)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        n = self.noise_config
        probs = [n['delete_probability'], n['class_flip_probability'], n['coordinate_shift_probability'],
                 n['resize_probability'], n['duplicate_probability']]
        if any(not (0.0 <= p <= 1.0) for p in probs):
            raise ValueError("All noise probabilities must be within [0.0, 1.0].")
        if sum(probs[:4]) > 1.0:  # duplicate is an independent roll
            raise ValueError("Sum of primary corruption probabilities (delete, flip, shift, resize) cannot exceed 1.0.")

    def inject(self) -> Path:
        """Create a noisy copy of the dataset and return its path."""
        logger.info("Starting synthetic noise injection...")

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        shutil.copytree(self.dataset_path, self.output_dir, ignore=shutil.ignore_patterns('*.cache'))

        yaml_files = list(self.output_dir.glob('*.yaml'))
        if not yaml_files:
            raise FileNotFoundError(f"No .yaml file found in {self.output_dir}")
        yaml_path = yaml_files[0]
        set_yaml_root(yaml_path, self.output_dir)   # absolute path, so Ultralytics trains on THIS copy

        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        num_classes = cfg.get('nc', len(cfg.get('names', [])))

        _, labels_dir = YOLODatasetLoader(str(self.output_dir)).get_split_paths('train')
        label_files = sorted(labels_dir.glob("*.txt"))
        logger.info(f"Processing {len(label_files)} training label files...")

        for label_file in label_files:
            self._corrupt_file(label_file, num_classes)

        with open(self.output_dir / "corruption_manifest.json", 'w') as f:
            json.dump(self.manifest, f, indent=2)
        by_type = {}
        for e in self.manifest:
            by_type[e['type']] = by_type.get(e['type'], 0) + 1
        logger.info(f"Noise injection complete: {by_type}")
        return self.output_dir

    def _shift(self, box: Dict) -> Dict:
        n = self.noise_config
        out = box.copy()
        dx = self.rng.choice([-1, 1]) * self.np_rng.uniform(n['shift_min'], n['shift_amount']) * box['w']
        dy = self.rng.choice([-1, 1]) * self.np_rng.uniform(n['shift_min'], n['shift_amount']) * box['h']
        out['x'] = _clip_center(box['x'] + dx, box['w'])
        out['y'] = _clip_center(box['y'] + dy, box['h'])
        return out

    def _resize(self, box: Dict) -> Dict:
        n = self.noise_config
        out = box.copy()
        sx = 1.0 + self.rng.choice([-1, 1]) * self.np_rng.uniform(n['resize_min'], n['resize_amount'])
        sy = 1.0 + self.rng.choice([-1, 1]) * self.np_rng.uniform(n['resize_min'], n['resize_amount'])
        out['w'] = float(np.clip(box['w'] * sx, 0.01, 1.0))
        out['h'] = float(np.clip(box['h'] * sy, 0.01, 1.0))
        out['x'] = _clip_center(box['x'], out['w'])
        out['y'] = _clip_center(box['y'], out['h'])
        return out

    def _corrupt_file(self, label_file: Path, num_classes: int):
        """Apply random corruptions to a single label file."""
        with open(label_file, 'r') as f:
            lines = f.readlines()

        n = self.noise_config
        t1 = n['delete_probability']
        t2 = t1 + n['class_flip_probability']
        t3 = t2 + n['coordinate_shift_probability']
        t4 = t3 + n['resize_probability']
        rel_path = label_file.resolve().relative_to(self.output_dir.resolve()).as_posix()

        def fmt(b):
            return f"{b['cls']} {b['x']:.6f} {b['y']:.6f} {b['w']:.6f} {b['h']:.6f}\n"

        new_lines = []
        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                new_lines.append(line)
                continue
            try:
                cls, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                new_lines.append(line)
                continue

            original_box = {'cls': cls, 'x': x, 'y': y, 'w': w, 'h': h}
            new_box = original_box.copy()
            r = self.rng.random()
            corrupted = False

            if r < t1:
                self.manifest.append({'file': rel_path, 'line': line_idx, 'type': 'deleted',
                                      'original': original_box, 'result': None})
                continue          # a deleted box is gone, and gets no duplicate either
            elif r < t2:
                if num_classes > 1:
                    new_box['cls'] = self.rng.choice([c for c in range(num_classes) if c != cls])
                    corrupted = True
                    self.manifest.append({'file': rel_path, 'line': line_idx, 'type': 'class_flip',
                                          'original': original_box, 'result': new_box})
            elif r < t3:
                new_box = self._shift(original_box)
                corrupted = True
                self.manifest.append({'file': rel_path, 'line': line_idx, 'type': 'coordinate_shift',
                                      'original': original_box, 'result': new_box,
                                      'iou_with_original': round(_iou(original_box, new_box), 4)})
            elif r < t4:
                new_box = self._resize(original_box)
                corrupted = True
                self.manifest.append({'file': rel_path, 'line': line_idx, 'type': 'resize',
                                      'original': original_box, 'result': new_box,
                                      'iou_with_original': round(_iou(original_box, new_box), 4)})

            new_lines.append(fmt(new_box) if corrupted else line)

            if self.rng.random() < n['duplicate_probability']:
                new_lines.append(fmt(new_box))
                self.manifest.append({'file': rel_path, 'line': line_idx, 'type': 'duplicate',
                                      'original': original_box, 'result': new_box})

        with open(label_file, 'w') as f:
            f.writelines(new_lines)
