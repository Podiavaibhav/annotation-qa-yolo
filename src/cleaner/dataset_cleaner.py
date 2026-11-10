"""
Module 3: Dataset Cleaning Engine.
Generates a cleaned copy of a YOLO dataset without modifying the original.
"""
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import yaml

from src.dataset_loader import YOLODatasetLoader

logger = logging.getLogger(__name__)

class DatasetCleaner:
    def __init__(self, dataset_path: str, config: Dict, 
                 analysis_report_path: str = None, model_audit_path: str = None):
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.loader = YOLODatasetLoader(dataset_path)
        
        self.cleaning_mode = config.get('cleaning_mode', 'conservative')
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.cleaned_dataset_dir = self.output_dir / 'cleaned_dataset'
        self.cleaning_report_dir = self.output_dir / 'cleaning'
        
        self.cleaned_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.cleaning_report_dir.mkdir(parents=True, exist_ok=True)

        self.invalid_to_remove = set()
        self.suspicious_to_remove = set()
        self.manifest = []

        if analysis_report_path and Path(analysis_report_path).exists():
            self._load_analysis_report(analysis_report_path)
        if model_audit_path and Path(model_audit_path).exists():
            self._load_model_audit(model_audit_path)

    def _load_analysis_report(self, report_path: str):
        with open(report_path, 'r') as f:
            report = json.load(f)
        for item in report.get('invalid_details', []):
            self.invalid_to_remove.add((item.get('image'), item.get('line')))
        if self.cleaning_mode == 'aggressive':
            for item in report.get('suspicious_details', []):
                box_tuple = tuple(item.get('box', []))
                self.suspicious_to_remove.add((item.get('image'), box_tuple))

    def _load_model_audit(self, audit_path: str):
        if self.cleaning_mode != 'aggressive':
            return
        csv_path = Path(audit_path) / 'class_disagreements.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # Flag image and ground truth class for removal in aggressive mode
                self.suspicious_to_remove.add((row['image'], row['gt_class']))

    def _should_remove_box(self, image_name: str, line_num: int, box_coords: Tuple[float, ...], class_id: int) -> Tuple[bool, str]:
        if (image_name, line_num) in self.invalid_to_remove:
            return True, 'invalid_annotation'
        if self.cleaning_mode == 'aggressive':
            if (image_name, tuple(box_coords)) in self.suspicious_to_remove:
                return True, 'suspicious_annotation_aggressive'
            if (image_name, class_id) in self.suspicious_to_remove:
                return True, 'model_class_disagreement'
        return False, ''

    def clean(self) -> Dict:
        logger.info(f"Starting dataset cleaning in '{self.cleaning_mode}' mode...")
        self._copy_dataset_structure()
        
        pairs = self.loader.get_image_label_pairs('train')
        total_removed = 0
        total_kept = 0
        
        for img_path, label_path in pairs:
            # Read raw lines to preserve line numbers for malformed entries
            with open(label_path, 'r') as f:
                raw_lines = f.readlines()
            
            kept_lines = []
            for line_num, line in enumerate(raw_lines, 1):
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cid = int(parts[0])
                        box_coords = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                        remove, reason = self._should_remove_box(img_path.name, line_num, box_coords, cid)
                        if remove:
                            total_removed += 1
                            self.manifest.append({'image': img_path.name, 'line': line_num, 'box': box_coords, 'reason': reason})
                            continue
                    except ValueError:
                        # Malformed numeric values
                        total_removed += 1
                        self.manifest.append({'image': img_path.name, 'line': line_num, 'reason': 'malformed_numeric'})
                        continue
                else:
                    # Malformed format
                    total_removed += 1
                    self.manifest.append({'image': img_path.name, 'line': line_num, 'reason': 'invalid_format'})
                    continue
                
                kept_lines.append(line)
                total_kept += 1
                    
            cleaned_label_path = self.cleaned_dataset_dir / 'labels' / 'train' / label_path.name
            with open(cleaned_label_path, 'w') as f:
                f.writelines(kept_lines)
                            
        self._save_manifest()
        self._save_summary(total_removed, total_kept)
        self._validate_cleaned_dataset()
        logger.info(f"Cleaning complete. Removed {total_removed} annotations.")
        return {'removed': total_removed, 'kept': total_kept}

    def _copy_dataset_structure(self):
        shutil.copy(self.loader.yaml_path, self.cleaned_dataset_dir / self.loader.yaml_path.name)
        
        for split in ['train', 'val', 'test']:
            try:
                src_images, src_labels = self.loader.get_split_paths(split)
            except FileNotFoundError:
                continue
                
            dst_images = self.cleaned_dataset_dir / 'images' / split
            dst_labels = self.cleaned_dataset_dir / 'labels' / split
            dst_images.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)
            
            for img_file in src_images.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    shutil.copy(img_file, dst_images / img_file.name)
                    # Copy corresponding label if it exists
                    label_file = src_labels / (img_file.stem + '.txt')
                    if label_file.exists():
                        shutil.copy(label_file, dst_labels / label_file.name)
                        
        # Rewrite YAML paths to be relative to the new cleaned dataset root
        yaml_path = self.cleaned_dataset_dir / self.loader.yaml_path.name
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Update paths to point to the local images/ and labels/ directories
        cfg['path'] = '.'
        if 'train' in cfg: cfg['train'] = 'images/train'
        if 'val' in cfg: cfg['val'] = 'images/val'
        if 'test' in cfg: cfg['test'] = 'images/test'
        
        with open(yaml_path, 'w') as f:
            yaml.dump(cfg, f)

    def _save_manifest(self):
        with open(self.cleaning_report_dir / 'cleaning_manifest.json', 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def _save_summary(self, removed: int, kept: int):
        summary = {'metric': ['Total Original', 'Kept', 'Removed', 'Mode'], 
                   'value': [removed + kept, kept, removed, self.cleaning_mode]}
        pd.DataFrame(summary).to_csv(self.cleaning_report_dir / 'cleaning_summary.csv', index=False)

    def _validate_cleaned_dataset(self):
        logger.info("Validating cleaned dataset...")
        cleaned_loader = YOLODatasetLoader(str(self.cleaned_dataset_dir))
        errors = cleaned_loader.validate_dataset()
        if errors:
            logger.error(f"Cleaned dataset validation failed: {errors}")
        else:
            logger.info("Cleaned dataset validation passed.")
