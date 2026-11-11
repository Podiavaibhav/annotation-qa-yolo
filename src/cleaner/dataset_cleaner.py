"""
Module 3: Dataset Cleaning Engine.
Generates a cleaned copy of a YOLO dataset without modifying the original.

Modes (each includes everything in the one before it):
  conservative    remove invalid and malformed boxes, and exact duplicates (the later copy only)
  model_assisted  also use the model audit to fix boxes one by one:
                    relabel a box when a confident prediction disagrees on its class,
                    move/resize a box when a confident same-class prediction overlaps it badly,
                    add a box where a confident prediction has no label
  aggressive      also remove every box the analyzer called suspicious (tiny, extreme aspect
                  ratio, huge). This can delete real small objects, so use it knowingly.
"""
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from src.dataset_loader import YOLODatasetLoader
from src.yaml_utils import set_yaml_root

logger = logging.getLogger(__name__)

MODES = ('conservative', 'model_assisted', 'aggressive')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


def _iou(a, b) -> float:
    """IoU of two (cx, cy, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class DatasetCleaner:
    def __init__(self, dataset_path: str, config: Dict,
                 analysis_report_path: str = None, model_audit_path: str = None):
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.loader = YOLODatasetLoader(dataset_path)

        self.cleaning_mode = config.get('cleaning_mode', 'conservative')
        if self.cleaning_mode not in MODES:
            raise ValueError(f"cleaning_mode must be one of {MODES}, got '{self.cleaning_mode}'")
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.cleaned_dataset_dir = self.output_dir / 'cleaned_dataset'
        self.cleaning_report_dir = self.output_dir / 'cleaning'

        self.relabel_conf = config.get('clean_relabel_confidence', 0.6)
        self.replace_conf = config.get('clean_replace_confidence', 0.6)
        self.add_conf = config.get('clean_add_confidence', 0.7)
        self.add_overlap_iou = config.get('clean_add_overlap_iou', 0.5)

        # (image name, 1-based line number) -> ...
        self.invalid_to_remove = set()
        self.duplicates_to_remove = set()
        self.suspicious_to_remove = set()
        self._suspicious_by_box = set()          # fallback for reports without line numbers
        self.relabel: Dict[Tuple[str, int], int] = {}
        self.replace: Dict[Tuple[str, int], Tuple[float, float, float, float]] = {}
        self.add: Dict[str, List[Tuple]] = defaultdict(list)
        self.manifest = []

        if analysis_report_path and Path(analysis_report_path).exists():
            self._load_analysis_report(analysis_report_path)
        if model_audit_path and Path(model_audit_path).exists():
            self._load_model_audit(model_audit_path)

    # ------------------------------------------------------------------ loading the reports
    def _load_analysis_report(self, report_path: str):
        with open(report_path, 'r') as f:
            report = json.load(f)
        for item in report.get('invalid_details', []):
            self.invalid_to_remove.add((item.get('image'), item.get('line')))

        for item in report.get('suspicious_details', []):
            issue, image, line = item.get('issue'), item.get('image'), item.get('line')
            if issue == 'exact_duplicate':
                if line is not None:
                    self.duplicates_to_remove.add((image, line))
                else:
                    logger.warning("Report has no line number for a duplicate in %s; skipping it", image)
            elif issue == 'overlapping_different_class':
                continue            # which of the two is wrong is not knowable from the labels alone
            elif self.cleaning_mode == 'aggressive':
                if line is not None:
                    self.suspicious_to_remove.add((image, line))
                else:
                    self._suspicious_by_box.add((image, tuple(item.get('box', []))))

    def _load_model_audit(self, audit_path: str):
        if self.cleaning_mode == 'conservative':
            return
        audit = Path(audit_path)

        def read(name):
            path = audit / name
            return pd.read_csv(path).to_dict('records') if path.exists() else []

        skipped = 0
        for r in read('class_disagreements.csv'):
            if 'gt_line' not in r:
                skipped += 1
                continue
            if r['confidence'] >= self.relabel_conf:
                self.relabel[(r['image'], int(r['gt_line']))] = int(r['pred_class'])
        for r in read('localization_disagreements.csv'):
            if 'gt_line' not in r or 'pred_cx' not in r:
                skipped += 1
                continue
            if r['confidence'] >= self.replace_conf:
                self.replace[(r['image'], int(r['gt_line']))] = (
                    float(r['pred_cx']), float(r['pred_cy']), float(r['pred_w']), float(r['pred_h']))
        for r in read('missing_annotations.csv'):
            if 'pred_cx' not in r:
                skipped += 1
                continue
            if r['confidence'] >= self.add_conf:
                self.add[r['image']].append((int(r['pred_class']), float(r['pred_cx']), float(r['pred_cy']),
                                             float(r['pred_w']), float(r['pred_h'])))
        if skipped:
            logger.warning("%d audit rows had no box coordinates / line numbers and were ignored. "
                           "Re-run detection with the current detector.", skipped)

    # ------------------------------------------------------------------ cleaning
    def clean(self) -> Dict:
        logger.info(f"Starting dataset cleaning in '{self.cleaning_mode}' mode...")
        self._reset_output_dir()
        self.cleaning_report_dir.mkdir(parents=True, exist_ok=True)
        self._copy_dataset_structure()

        src_images, src_labels = self.loader.get_split_paths('train')
        counts = {'removed': 0, 'kept': 0, 'relabeled': 0, 'replaced': 0, 'added': 0}

        for img_file in sorted(src_images.glob('*')):
            if img_file.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = src_labels / (img_file.stem + '.txt')
            raw_lines = label_path.read_text().splitlines(keepends=True) if label_path.exists() else []
            name = img_file.name

            kept_lines, kept_boxes = [], []
            for line_num, line in enumerate(raw_lines, 1):
                text = line.strip()
                if not text:
                    continue
                parts = text.split()
                if len(parts) != 5:
                    self._record_removed(counts, name, line_num, 'invalid_format')
                    continue
                try:
                    cid = int(parts[0])
                    box = tuple(float(p) for p in parts[1:])
                except ValueError:
                    self._record_removed(counts, name, line_num, 'malformed_numeric')
                    continue

                key = (name, line_num)
                reason = None
                if key in self.invalid_to_remove:
                    reason = 'invalid_annotation'
                elif key in self.duplicates_to_remove:
                    reason = 'exact_duplicate'
                elif self.cleaning_mode == 'aggressive' and (
                        key in self.suspicious_to_remove or (name, box) in self._suspicious_by_box):
                    reason = 'suspicious_annotation_aggressive'
                if reason:
                    self._record_removed(counts, name, line_num, reason, box)
                    continue

                new_cid, new_box = cid, box
                if key in self.relabel:
                    new_cid = self.relabel[key]
                    counts['relabeled'] += 1
                    self.manifest.append({'image': name, 'line': line_num, 'action': 'relabeled',
                                          'from_class': cid, 'to_class': new_cid})
                if key in self.replace:
                    new_box = self.replace[key]
                    counts['replaced'] += 1
                    self.manifest.append({'image': name, 'line': line_num, 'action': 'box_replaced',
                                          'from_box': box, 'to_box': new_box})
                changed = (new_cid, new_box) != (cid, box)
                kept_lines.append(self._format(new_cid, new_box) if changed else text + "\n")
                kept_boxes.append(new_box)
                counts['kept'] += 1

            for cls, *abox in self.add.get(name, []):
                abox = tuple(abox)
                if any(_iou(abox, kb) >= self.add_overlap_iou for kb in kept_boxes):
                    continue        # something already labels this object
                kept_lines.append(self._format(cls, abox))
                kept_boxes.append(abox)
                counts['added'] += 1
                self.manifest.append({'image': name, 'action': 'box_added', 'class': cls, 'box': abox})

            out_label = self.cleaned_dataset_dir / 'labels' / 'train' / (img_file.stem + '.txt')
            with open(out_label, 'w') as f:
                f.writelines(kept_lines)

        self._save_manifest()
        self._save_summary(counts)
        self._validate_cleaned_dataset()
        logger.info("Cleaning complete: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        return counts

    @staticmethod
    def _format(cls, box) -> str:
        return f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"

    def _record_removed(self, counts, name, line_num, reason, box=None):
        counts['removed'] += 1
        entry = {'image': name, 'line': line_num, 'action': 'removed', 'reason': reason}
        if box is not None:
            entry['box'] = box
        self.manifest.append(entry)

    def _reset_output_dir(self):
        """Start from an empty cleaned dataset so files from an earlier run cannot linger."""
        if not self.cleaned_dataset_dir.exists():
            return
        src = self.loader.dataset_dir.resolve()
        dst = self.cleaned_dataset_dir.resolve()
        if dst == src or dst in src.parents:
            raise ValueError(f"Refusing to delete {dst}: it contains the source dataset.")
        shutil.rmtree(dst)

    def _copy_dataset_structure(self):
        copied = set()
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
                if img_file.suffix.lower() in IMAGE_SUFFIXES:
                    shutil.copy(img_file, dst_images / img_file.name)
                    label_file = src_labels / (img_file.stem + '.txt')
                    if label_file.exists():
                        shutil.copy(label_file, dst_labels / label_file.name)
            copied.add(split)

        # The cleaned copy always uses images/<split> + labels/<split>, whatever the source layout was.
        with open(self.loader.yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        for split in copied:
            cfg[split] = f'images/{split}'
        if 'val' in copied:
            cfg.pop('valid', None)
            cfg.pop('validation', None)
        yaml_path = self.cleaned_dataset_dir / self.loader.yaml_path.name
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        set_yaml_root(yaml_path, self.cleaned_dataset_dir)   # absolute, so Ultralytics reads THIS copy

    def _save_manifest(self):
        with open(self.cleaning_report_dir / 'cleaning_manifest.json', 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def _save_summary(self, counts: Dict):
        rows = [('Total original', counts['removed'] + counts['kept']), ('Kept', counts['kept']),
                ('Removed', counts['removed']), ('Relabeled', counts['relabeled']),
                ('Boxes replaced', counts['replaced']), ('Boxes added', counts['added']),
                ('Mode', self.cleaning_mode)]
        pd.DataFrame({'metric': [r[0] for r in rows], 'value': [r[1] for r in rows]}).to_csv(
            self.cleaning_report_dir / 'cleaning_summary.csv', index=False)

    def _validate_cleaned_dataset(self):
        logger.info("Validating cleaned dataset...")
        errors = YOLODatasetLoader(str(self.cleaned_dataset_dir)).validate_dataset()
        if errors:
            logger.error(f"Cleaned dataset validation failed: {errors}")
        else:
            logger.info("Cleaned dataset validation passed.")
