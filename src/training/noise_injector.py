"""
Synthetic Noise Injection Utility.
Corrupts training labels to simulate annotation errors.
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
    """Injects controlled noise into a YOLO dataset for robustness testing."""

    def __init__(self, dataset_path: str, config: Dict):
        path = Path(dataset_path)
        self.dataset_path = path.parent if path.is_file() else path
        self.config = config
        self.noise_config = config.get('noise', {})
        self.output_dir = Path(config.get('output_dir', 'outputs')) / 'noisy_dataset'
        self.manifest = []
        
        # Use dedicated random generators to avoid modifying global state
        seed = config.get('seed', 42)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        
        # Validate probabilities
        p_del = self.noise_config.get('delete_probability', 0.05)
        p_flip = self.noise_config.get('class_flip_probability', 0.03)
        p_shift = self.noise_config.get('coordinate_shift_probability', 0.10)
        p_resize = self.noise_config.get('resize_probability', 0.20)
        p_dup = self.noise_config.get('duplicate_probability', 0.02)
        
        probs = [p_del, p_flip, p_shift, p_resize, p_dup]
        if any(not (0.0 <= p <= 1.0) for p in probs):
            raise ValueError("All noise probabilities must be within [0.0, 1.0].")
        if sum(probs[:4]) > 1.0:  # Exclude duplicate as it's an independent roll
            raise ValueError("Sum of primary corruption probabilities (delete, flip, shift, resize) cannot exceed 1.0.")

    def inject(self) -> Path:
        """Create a noisy copy of the dataset and return its path."""
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
        """Apply random corruptions to a single label file."""
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
            
            # Calculate dynamic cumulative thresholds from config
            p_del = self.noise_config.get('delete_probability', 0.05)
            p_flip = self.noise_config.get('class_flip_probability', 0.03)
            p_shift = self.noise_config.get('coordinate_shift_probability', 0.10)
            p_resize = self.noise_config.get('resize_probability', 0.20)
            p_dup = self.noise_config.get('duplicate_probability', 0.02)
            
            t1 = p_del
            t2 = t1 + p_flip
            t3 = t2 + p_shift
            t4 = t3 + p_resize
            
            r = self.rng.random()
            corrupted = False
            is_deleted = False
            new_box = original_box.copy()
            
            # FIX: Use self.output_dir as the base for relative paths
            rel_path = str(label_file.relative_to(self.output_dir))

            if r < t1:
                is_deleted = True
                corrupted = True
                self.manifest.append({
                    'file': rel_path, 'line': line_idx, 'type': 'deleted', 
                    'original': original_box, 'result': None
                })
            elif r < t2:
                if num_classes > 1:
                    new_cls = self.rng.choice([c for c in range(num_classes) if c != cls])
                    new_box['cls'] = new_cls
                    corrupted = True
                    self.manifest.append({
                        'file': rel_path, 'line': line_idx, 'type': 'class_flip', 
                        'original': original_box, 'result': new_box
                    })
                else:
                    # Single class: skip corruption safely
                    pass
            elif r < t3:
                shift = self.noise_config.get('shift_amount', 0.1)
                new_box['x'] = float(np.clip(x + self.np_rng.uniform(-shift, shift) * w, 0.0, 1.0))
                new_box['y'] = float(np.clip(y + self.np_rng.uniform(-shift, shift) * h, 0.0, 1.0))
                corrupted = True
                self.manifest.append({
                    'file': rel_path, 'line': line_idx, 'type': 'coordinate_shift', 
                    'original': original_box, 'result': new_box
                })
            elif r < t4:
                resize = self.noise_config.get('resize_amount', 0.2)
                scale = 1.0 + self.np_rng.uniform(-resize, resize)
                new_box['w'] = float(np.clip(w * scale, 0.01, 1.0))
                new_box['h'] = float(np.clip(h * scale, 0.01, 1.0))
                corrupted = True
                self.manifest.append({
                    'file': rel_path, 'line': line_idx, 'type': 'resize', 
                    'original': original_box, 'result': new_box
                })

            if is_deleted:
                continue
            elif not corrupted:
                new_lines.append(line)
            else:
                new_lines.append(f"{new_box['cls']} {new_box['x']:.6f} {new_box['y']:.6f} {new_box['w']:.6f} {new_box['h']:.6f}\n")

            if self.rng.random() < p_dup:
                new_lines.append(f"{new_box['cls']} {new_box['x']:.6f} {new_box['y']:.6f} {new_box['w']:.6f} {new_box['h']:.6f}\n")
                self.manifest.append({
                    'file': rel_path, 'line': line_idx, 'type': 'duplicate', 
                    'original': original_box, 'result': new_box
                })

        with open(label_file, 'w') as f:
            f.writelines(new_lines)
