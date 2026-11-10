"""
Module 2: Model-Assisted Annotation Error Detection.
Compares YOLOv8 predictions against ground-truth annotations to identify potential errors.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from src.dataset_loader import YOLODatasetLoader

logger = logging.getLogger(__name__)


class ModelDisagreementDetector:
    """
    Detects annotation errors by comparing model predictions with ground truth.
    """

    def __init__(self, dataset_path: str, config: Dict, model_name: str = "yolov8n.pt"):
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.loader = YOLODatasetLoader(dataset_path)
        self.model_name = model_name
        
        self.conf_thresh = config.get('confidence_threshold', 0.50)
        self.iou_thresh = config.get('iou_threshold', 0.50)
        
        self.disagreements = []
        self.missing_annotations = []
        self.localization_disagreements = []
        
        self.output_dir = Path(config.get('output_dir', 'outputs')) / 'model_audit'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'flagged_images').mkdir(exist_ok=True)

    def _calculate_iou_matrix(self, gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
        """Calculate IoU matrix between ground truth and predicted boxes (xyxy format)."""
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            return np.zeros((len(gt_boxes), len(pred_boxes)))
            
        x1 = np.maximum(gt_boxes[:, 0:1], pred_boxes[:, 0:1].T)
        y1 = np.maximum(gt_boxes[:, 1:2], pred_boxes[:, 1:2].T)
        x2 = np.minimum(gt_boxes[:, 2:3], pred_boxes[:, 2:3].T)
        y2 = np.minimum(gt_boxes[:, 3:4], pred_boxes[:, 3:4].T)

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        
        union = area_gt[:, None] + area_pred[None, :] - intersection
        
        return intersection / np.maximum(union, 1e-6)

    def _greedy_match(self, iou_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
        """Perform greedy one-to-one matching based on IoU."""
        matches = []
        used_gt = set()
        used_pred = set()
        
        # Flatten and sort indices by IoU descending
        indices = np.argsort(iou_matrix, axis=None)[::-1]
        
        for idx in indices:
            i, j = np.unravel_index(idx, iou_matrix.shape)
            if iou_matrix[i, j] < self.iou_thresh:
                break
            if i not in used_gt and j not in used_pred:
                matches.append((i, j, iou_matrix[i, j]))
                used_gt.add(i)
                used_pred.add(j)
                
        return matches

    def _yolo_to_xyxy(self, ann: Dict, img_w: int, img_h: int) -> List[float]:
        """Convert normalized YOLO xywh to absolute xyxy."""
        x_center = ann['x_center'] * img_w
        y_center = ann['y_center'] * img_h
        w = ann['width'] * img_w
        h = ann['height'] * img_h
        
        return [
            x_center - w / 2, 
            y_center - h / 2, 
            x_center + w / 2, 
            y_center + h / 2
        ]

    def _draw_and_save(self, img_path: Path, gt_boxes: List, pred_boxes: List, matches: List, filename: str):
        """Draw GT (green) and Predictions (red) on image and save."""
        img = cv2.imread(str(img_path))
        if img is None:
            return
            
        for box in gt_boxes:
            x1, y1, x2, y2 = [int(b) for b in box[:4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"GT:{box[4]}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        for box in pred_boxes:
            x1, y1, x2, y2 = [int(b) for b in box[:4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"Pred:{box[4]}:{box[5]:.2f}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        cv2.imwrite(str(self.output_dir / 'flagged_images' / filename), img)

    def detect(self) -> Dict:
        """Run model inference and compare with ground truth."""
        logger.info(f"Loading model: {self.model_name}")
        model = YOLO(self.model_name)
        
        pairs = self.loader.get_image_label_pairs('train')
        logger.info(f"Processing {len(pairs)} training images...")
        
        for img_path, label_path in pairs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            
            # Load GT
            gt_anns = self.loader.load_annotations(label_path)
            gt_boxes_abs = []
            for ann in gt_anns:
                box = self._yolo_to_xyxy(ann, img_w, img_h)
                box.append(ann['class_id'])
                gt_boxes_abs.append(box)
            
            # Run inference
            results = model(str(img_path), verbose=False, conf=self.conf_thresh)
            pred_boxes_abs = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                    conf = boxes.conf[i].cpu().numpy().item()
                    cls_id = int(boxes.cls[i].cpu().numpy().item())
                    pred_boxes_abs.append(xyxy + [cls_id, conf])
            
            # Matching
            gt_arr = np.array([b[:4] for b in gt_boxes_abs]) if gt_boxes_abs else np.empty((0, 4))
            pred_arr = np.array([b[:4] for b in pred_boxes_abs]) if pred_boxes_abs else np.empty((0, 4))
            
            iou_matrix = self._calculate_iou_matrix(gt_arr, pred_arr)
            matches = self._greedy_match(iou_matrix)
            
            matched_gt = {m[0] for m in matches}
            matched_pred = {m[1] for m in matches}
            
            is_flagged = False
            
            # Check matched pairs
            for gt_idx, pred_idx, iou in matches:
                gt_cls = gt_boxes_abs[gt_idx][4]
                pred_cls = pred_boxes_abs[pred_idx][4]
                pred_conf = pred_boxes_abs[pred_idx][5]
                
                if gt_cls != pred_cls:
                    self.disagreements.append({
                        'image': img_path.name, 'gt_class': gt_cls, 'pred_class': pred_cls,
                        'confidence': pred_conf, 'iou': iou, 'type': 'class_disagreement'
                    })
                    is_flagged = True
                elif iou < 0.7: # Localization threshold
                    self.localization_disagreements.append({
                        'image': img_path.name, 'class': gt_cls,
                        'confidence': pred_conf, 'iou': iou, 'type': 'localization_disagreement'
                    })
                    is_flagged = True
                    
            # Check unmatched predictions (potential missing GT)
            for pred_idx in range(len(pred_boxes_abs)):
                if pred_idx not in matched_pred:
                    self.missing_annotations.append({
                        'image': img_path.name, 'pred_class': pred_boxes_abs[pred_idx][4],
                        'confidence': pred_boxes_abs[pred_idx][5], 'type': 'missing_annotation'
                    })
                    is_flagged = True
                    
            # Save visualization if flagged
            if is_flagged:
                self._draw_and_save(img_path, gt_boxes_abs, pred_boxes_abs, matches, img_path.name)
                
        report = self._generate_report()
        self._save_csvs()
        logger.info(f"Detection complete. Reports saved to {self.output_dir}")
        return report

    def _generate_report(self) -> Dict:
        return {
            'total_disagreements': len(self.disagreements),
            'total_missing': len(self.missing_annotations),
            'total_localization_issues': len(self.localization_disagreements)
        }

    def _save_csvs(self):
        if self.disagreements:
            pd.DataFrame(self.disagreements).to_csv(self.output_dir / 'class_disagreements.csv', index=False)
        if self.missing_annotations:
            pd.DataFrame(self.missing_annotations).to_csv(self.output_dir / 'missing_annotations.csv', index=False)
        if self.localization_disagreements:
            pd.DataFrame(self.localization_disagreements).to_csv(self.output_dir / 'localization_disagreements.csv', index=False)
