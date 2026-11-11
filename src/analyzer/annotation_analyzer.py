"""
Module 1: Dataset Quality Analyzer.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.dataset_loader import YOLODatasetLoader

logger = logging.getLogger(__name__)

class AnnotationAnalyzer:
    def __init__(self, dataset_path: str, config: Dict):
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.loader = YOLODatasetLoader(dataset_path)
        self.suspicious_annotations = []
        self.invalid_annotations = []
        self.class_counts = {}
        self.box_areas = []
        self.output_dir = Path(config.get('output_dir', 'outputs')) / 'analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        x1_a, y1_a = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        x2_a, y2_a = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        x1_b, y1_b = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
        x2_b, y2_b = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
        x_inter = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
        y_inter = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
        intersection = x_inter * y_inter
        area_a = box1[2] * box1[3]
        area_b = box2[2] * box2[3]
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _validate_raw_format(self, label_path: Path) -> List[Dict]:
        issues = []
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) != 5:
                    issues.append({'image': label_path.name, 'line': line_num, 'issue': 'invalid_format', 'details': f'Expected 5 values, got {len(parts)}'})
                    continue
                try:
                    float(parts[1]); float(parts[2]); float(parts[3]); float(parts[4])
                except ValueError:
                    issues.append({'image': label_path.name, 'line': line_num, 'issue': 'non_numeric', 'details': f'Non-numeric values in line: {line}'})
        return issues

    def analyze(self) -> Dict:
        logger.info("Starting dataset analysis...")
        pairs = self.loader.get_image_label_pairs('train')
        num_classes = self.loader.num_classes
        class_names = self.loader.classes
        tiny_thresh = self.config.get('tiny_box_threshold', 0.001)
        dup_thresh = self.config.get('duplicate_iou_threshold', 0.98)
        max_area = self.config.get('max_box_area', 0.95)
        min_ar = self.config.get('min_aspect_ratio', 0.1)
        max_ar = self.config.get('max_aspect_ratio', 10.0)

        for img_path, label_path in pairs:
            self.invalid_annotations.extend(self._validate_raw_format(label_path))
            annotations = self.loader.load_annotations(label_path)
            valid_boxes = []
            
            for idx, ann in enumerate(annotations):
                cid, x, y, w, h = ann['class_id'], ann['x_center'], ann['y_center'], ann['width'], ann['height']
                self.class_counts[cid] = self.class_counts.get(cid, 0) + 1
                area = w * h
                self.box_areas.append(area)
                is_invalid = False
                
                if w <= 0 or h <= 0:
                    self.invalid_annotations.append({'image': img_path.name, 'line': ann['line_number'], 'issue': 'zero_or_negative_dim', 'details': f'w={w}, h={h}'})
                    is_invalid = True
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    self.invalid_annotations.append({'image': img_path.name, 'line': ann['line_number'], 'issue': 'out_of_bounds', 'details': f'x={x}, y={y}'})
                    is_invalid = True
                if cid < 0 or cid >= num_classes:
                    self.invalid_annotations.append({'image': img_path.name, 'line': ann['line_number'], 'issue': 'invalid_class_id', 'details': f'class_id={cid}'})
                    is_invalid = True

                if not is_invalid:
                    if area < tiny_thresh:
                        self.suspicious_annotations.append({'image': img_path.name, 'class_id': cid, 'box': [x, y, w, h], 'line': ann['line_number'], 'issue': 'tiny_box', 'severity': 'warning', 'details': f'area={area:.5f}'})
                    aspect_ratio = w / h if h > 0 else 0
                    if aspect_ratio < min_ar or aspect_ratio > max_ar:
                        self.suspicious_annotations.append({'image': img_path.name, 'class_id': cid, 'box': [x, y, w, h], 'line': ann['line_number'], 'issue': 'extreme_aspect_ratio', 'severity': 'warning', 'details': f'ar={aspect_ratio:.2f}'})
                    if area > max_area:
                        self.suspicious_annotations.append({'image': img_path.name, 'class_id': cid, 'box': [x, y, w, h], 'line': ann['line_number'], 'issue': 'unusually_large', 'severity': 'warning', 'details': f'area={area:.3f}'})
                    valid_boxes.append((idx, [x, y, w, h], cid, ann['line_number']))

            # Duplicates: same class = exact_duplicate, different class = overlapping_different_class
            for i in range(len(valid_boxes)):
                for j in range(i + 1, len(valid_boxes)):
                    idx_i, box_i, cls_i, line_i = valid_boxes[i]
                    idx_j, box_j, cls_j, line_j = valid_boxes[j]
                    iou = self._calculate_iou(box_i, box_j)
                    if iou >= dup_thresh:
                        issue_type = 'exact_duplicate' if cls_i == cls_j else 'overlapping_different_class'
                        self.suspicious_annotations.append({
                            'image': img_path.name, 'class_id': cls_j, 'box': box_j, 'line': line_j,
                            'issue': issue_type, 'severity': 'warning', 'details': f'IoU={iou:.3f}, other_class={cls_i}'
                        })

        report = self._generate_report(len(pairs))
        self._save_visualizations(class_names)
        self._save_csv_reports()
        logger.info(f"Analysis complete. Reports saved to {self.output_dir}")
        return report

    def _generate_report(self, total_images: int) -> Dict:
        total_annotations = sum(self.class_counts.values())
        # format errors never made it into class_counts, and one box can have several issues
        bad_boxes = {(i['image'], i['line']) for i in self.invalid_annotations
                     if i['issue'] not in ('invalid_format', 'non_numeric')}
        valid_annotations = total_annotations - len(bad_boxes)
        counts = list(self.class_counts.values()) if self.class_counts else [0]
        return {
            'summary': {'total_images': total_images, 'total_annotations': total_annotations, 'valid_annotations': valid_annotations,
                        'invalid_annotations': len(self.invalid_annotations), 'suspicious_annotations': len(self.suspicious_annotations),
                        'class_imbalance_ratio': max(counts) / min(counts) if min(counts) > 0 else 0},
            'class_distribution': {str(k): v for k, v in self.class_counts.items()},
            'invalid_details': self.invalid_annotations,
            'suspicious_details': self.suspicious_annotations
        }

    def _save_visualizations(self, class_names: List[str]):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        if self.class_counts:
            classes = [class_names[i] if i < len(class_names) else str(i) for i in self.class_counts.keys()]
            sns.barplot(x=classes, y=list(self.class_counts.values()))
        plt.title('Class Distribution'); plt.xlabel('Class'); plt.ylabel('Count'); plt.xticks(rotation=45)
        plt.tight_layout(); plt.savefig(self.output_dir / 'class_distribution.png', dpi=150); plt.close()

        plt.figure(figsize=(10, 6))
        if self.box_areas:
            sns.histplot(self.box_areas, bins=50, kde=True)
        plt.title('Bounding Box Area Distribution'); plt.xlabel('Normalized Area'); plt.ylabel('Count')
        plt.tight_layout(); plt.savefig(self.output_dir / 'box_area_distribution.png', dpi=150); plt.close()

    def _save_csv_reports(self):
        if self.invalid_annotations:
            pd.DataFrame(self.invalid_annotations).to_csv(self.output_dir / 'invalid_annotations.csv', index=False)
        if self.suspicious_annotations:
            pd.DataFrame(self.suspicious_annotations).to_csv(self.output_dir / 'suspicious_annotations.csv', index=False)
