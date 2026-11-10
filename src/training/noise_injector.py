"""
Synthetic Noise Injection Utility.
"""
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

logger = logging.getLogger(__name__)

class NoiseInjector:
    def __init__(self, dataset_path: str, config: Dict):
        path = Path(dataset_path)
        self.dataset_path = path.parent if path.is_file() else path
        self.config = config
        self.noise_config = config.get('noise', {})
        self.output_dir = Path(config.get('output_dir', 'outputs')) / 'noisy_dataset'
        self.manifest = []
        
        # Seed random generators for reproducibility
        seed = config.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)

    def inject(self) -> Path:
        logger.info("Starting synthetic noise injection...")
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        shutil.copytree(self.dataset_path, self.output_dir)

        yaml_files = list(self.output_dir.glob('*.yaml'))
        if not yaml_files:
            raise FileNotFoundError(f"No .yaml file found in {self.output_dir}")
        yaml_path = yaml_files[0]

        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        num_classes = cfg.get('nc', len(cfg.get('names', [])))

        labels_dir = self.output_dir / "labels" / "train"
        if not labels_dir.exists():
            labels_dir = self.output_dir / "train" / "labels"

        label_files = list(labels_dir.glob("*.txt"))
        logger.info(f"Processing {len(label_files)} training label files...")

        for label_file in label_files:
            self._corrupt_file(label_file, num_classes)

        with open(self.output_dir / "corruption_manifest.json", 'w') as f:
            json.dump(self.manifest, f, indent=2)
        logger.info(f"Noise injection complete. Manifest saved.")
        return self.output_dir

    def _corrupt_file(self, label_file: Path, num_classes: int):
        with open(label_file, 'r') as f:
            lines = f.readlines()

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
            r = random.random()
            corrupted = False
            is_deleted = False

            if r < self.noise_config.get('delete_probability', 0.05):
                self.manifest.append({'file': label_file.name, 'line': line_idx, 'type': 'deleted', 'original': original_box})
                is_deleted = True
                corrupted = True
            elif r < self.noise_config.get('delete_probability', 0.05) + self.noise_config.get('class_flip_probability', 0.03):
                if num_classes > 1:
                    new_cls = random.choice([c for c in range(num_classes) if c != cls])
                else:
                    new_cls = cls # Cannot flip if only 1 class
                self.manifest.append({'file': label_file.name, 'line': line_idx, 'type': 'class_flip', 'original': original_box, 'new_cls': new_cls})
                cls = new_cls
                corrupted = True
            elif r < 0.08 + self.noise_config.get('coordinate_shift_probability', 0.10):
                shift = self.noise_config.get('shift_amount', 0.1)
                x = float(np.clip(x + random.uniform(-shift, shift) * w, 0, 1))
                y = float(np.clip(y + random.uniform(-shift, shift) * h, 0, 1))
                self.manifest.append({'file': label_file.name, 'line': line_idx, 'type': 'coordinate_shift', 'original': original_box, 'new': {'x': x, 'y': y}})
                corrupted = True
            elif r < 0.18 + self.noise_config.get('resize_probability', 0.20):
                resize = self.noise_config.get('resize_amount', 0.2)
                scale = 1 + random.uniform(-resize, resize)
                w = float(np.clip(w * scale, 0.01, 1))
                h = float(np.clip(h * scale, 0.01, 1))
                self.manifest.append({'file': label_file.name, 'line': line_idx, 'type': 'resize', 'original': original_box, 'new': {'w': w, 'h': h}})
                corrupted = True

            if is_deleted:
                continue
            elif not corrupted:
                new_lines.append(line)
            else:
                new_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            if random.random() < self.noise_config.get('duplicate_probability', 0.02):
                new_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                self.manifest.append({'file': label_file.name, 'line': line_idx, 'type': 'duplicate', 'original': original_box})

        with open(label_file, 'w') as f:
            f.writelines(new_lines)
