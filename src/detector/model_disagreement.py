"""
Module 2: Model-Assisted Annotation Error Detection.
Compares YOLOv8 predictions against ground-truth annotations to identify potential errors.

Every flag records the box coordinates and the label's line number, so the cleaner can fix the
exact box instead of guessing from an image name.

Which model predicts matters. A model that trained on an image has seen its (possibly wrong)
labels and tends to repeat them. Pass `fold_models` (image file name -> weights path, see
oof_models.py) so each image is predicted by a model that never trained on it. Without it, a
single model (`model_name`) predicts every image, which is only sensible for a model trained
elsewhere on the same classes.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.dataset_loader import YOLODatasetLoader

logger = logging.getLogger(__name__)


class ModelDisagreementDetector:
    """Detects annotation errors by comparing model predictions with ground truth."""

    def __init__(self, dataset_path: str, config: Dict, model_name: str = "yolov8n.pt",
                 fold_models: Optional[Dict[str, str]] = None):
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.loader = YOLODatasetLoader(dataset_path)
        self.model_name = model_name
        self.fold_models = fold_models or {}
        self._models = {}

        self.conf_thresh = config.get('confidence_threshold', 0.50)
        self.iou_thresh = config.get('iou_threshold', 0.50)
        self.loc_iou_thresh = config.get('localization_iou_threshold', 0.70)  # matched but below this = poorly placed
        self.loc_min_iou = config.get('localization_min_iou', 0.10)           # unmatched pair above this = same object

        self.disagreements: List[Dict] = []
        self.missing_annotations: List[Dict] = []
        self.localization_disagreements: List[Dict] = []
        self.unmatched_labels: List[Dict] = []

        self.output_dir = Path(config.get('output_dir', 'outputs')) / 'model_audit'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'flagged_images').mkdir(exist_ok=True)

    # ------------------------------------------------------------------ geometry
    def _calculate_iou_matrix(self, gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
        """IoU matrix between ground truth and predicted boxes (xyxy format)."""
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
        """Greedy one-to-one matching, best IoU first, stopping below the IoU threshold."""
        matches, used_gt, used_pred = [], set(), set()
        if iou_matrix.size == 0:
            return matches
        for idx in np.argsort(iou_matrix, axis=None)[::-1]:
            i, j = np.unravel_index(idx, iou_matrix.shape)
            if iou_matrix[i, j] < self.iou_thresh:
                break
            if i not in used_gt and j not in used_pred:
                matches.append((int(i), int(j), float(iou_matrix[i, j])))
                used_gt.add(i)
                used_pred.add(j)
        return matches

    def _yolo_to_xyxy(self, ann: Dict, img_w: int, img_h: int) -> List[float]:
        """Normalized YOLO xywh -> absolute xyxy."""
        x_center, y_center = ann['x_center'] * img_w, ann['y_center'] * img_h
        w, h = ann['width'] * img_w, ann['height'] * img_h
        return [x_center - w / 2, y_center - h / 2, x_center + w / 2, y_center + h / 2]

    @staticmethod
    def _xyxy_to_norm(xyxy: List[float], img_w: int, img_h: int) -> List[float]:
        """Absolute xyxy -> normalized (cx, cy, w, h)."""
        x1, y1, x2, y2 = xyxy
        return [(x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h]

    # ------------------------------------------------------------------ models
    def _model_for(self, image_name: str):
        key = self.fold_models.get(image_name)
        if key is None:
            if self.fold_models:
                raise KeyError(f"No out-of-fold model recorded for {image_name}")
            key = self.model_name
        if key not in self._models:
            logger.info(f"Loading model: {key}")
            self._models[key] = YOLO(key)
        return self._models[key]

    def _predict(self, img_path: Path) -> List[List[float]]:
        results = self._model_for(img_path.name)(str(img_path), verbose=False, conf=self.conf_thresh)
        preds = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = boxes.conf[i].cpu().numpy().item()
                cls_id = int(boxes.cls[i].cpu().numpy().item())
                preds.append(xyxy + [cls_id, conf])
        return preds

    # ------------------------------------------------------------------ drawing
    def _draw_and_save(self, img_path: Path, gt_boxes: List, pred_boxes: List, filename: str):
        """GT in green, predictions in red."""
        img = cv2.imread(str(img_path))
        if img is None:
            return
        for box in gt_boxes:
            x1, y1, x2, y2 = [int(b) for b in box[:4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"GT:{box[4]}", (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for box in pred_boxes:
            x1, y1, x2, y2 = [int(b) for b in box[:4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"Pred:{box[4]}:{box[5]:.2f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(self.output_dir / 'flagged_images' / filename), img)

    # ------------------------------------------------------------------ main loop
    def detect(self) -> Dict:
        """Run model inference and compare with ground truth."""
        pairs = self.loader.get_image_label_pairs('train')
        logger.info(f"Processing {len(pairs)} training images...")
        self.disagreements, self.missing_annotations = [], []
        self.localization_disagreements, self.unmatched_labels = [], []

        for img_path, label_path in pairs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            gt = []
            for ann in self.loader.load_annotations(label_path):
                xyxy = self._yolo_to_xyxy(ann, img_w, img_h)
                gt.append({'line': ann['line_number'], 'cls': ann['class_id'], 'xyxy': xyxy,
                           'norm': [ann['x_center'], ann['y_center'], ann['width'], ann['height']]})

            preds = []
            for p in self._predict(img_path):
                preds.append({'xyxy': p[:4], 'cls': int(p[4]), 'conf': float(p[5]),
                              'norm': self._xyxy_to_norm(p[:4], img_w, img_h)})

            gt_arr = np.array([g['xyxy'] for g in gt]) if gt else np.empty((0, 4))
            pred_arr = np.array([p['xyxy'] for p in preds]) if preds else np.empty((0, 4))
            iou_matrix = self._calculate_iou_matrix(gt_arr, pred_arr)
            matches = self._greedy_match(iou_matrix)

            matched_gt = {m[0] for m in matches}
            matched_pred = {m[1] for m in matches}
            flagged = False

            def gt_cols(g):
                return dict(zip(['gt_cx', 'gt_cy', 'gt_w', 'gt_h'], g['norm']))

            def pred_cols(p):
                return dict(zip(['pred_cx', 'pred_cy', 'pred_w', 'pred_h'], p['norm']))

            for gi, pi, iou in matches:
                g, p = gt[gi], preds[pi]
                if g['cls'] != p['cls']:
                    self.disagreements.append({
                        'image': img_path.name, 'gt_line': g['line'], 'gt_class': g['cls'], 'pred_class': p['cls'],
                        'confidence': p['conf'], 'iou': iou, **gt_cols(g), **pred_cols(p), 'type': 'class_disagreement'})
                    flagged = True
                elif iou < self.loc_iou_thresh:
                    self.localization_disagreements.append({
                        'image': img_path.name, 'gt_line': g['line'], 'class': g['cls'], 'confidence': p['conf'],
                        'iou': iou, **gt_cols(g), **pred_cols(p), 'type': 'localization_disagreement'})
                    flagged = True

            # A label that overlaps a same-class prediction only a little is the same object, badly placed.
            # (Matching requires IoU >= iou_thresh, so a badly shifted box would otherwise look like one
            # spurious label plus one missing annotation.)
            free_gt = [i for i in range(len(gt)) if i not in matched_gt]
            free_pred = {j for j in range(len(preds)) if j not in matched_pred}
            for gi in free_gt:
                best, best_iou = None, self.loc_min_iou
                for pj in free_pred:
                    if preds[pj]['cls'] != gt[gi]['cls']:
                        continue
                    if iou_matrix[gi, pj] >= best_iou:
                        best, best_iou = pj, float(iou_matrix[gi, pj])
                g = gt[gi]
                if best is not None:
                    p = preds[best]
                    free_pred.discard(best)
                    matched_gt.add(gi)
                    self.localization_disagreements.append({
                        'image': img_path.name, 'gt_line': g['line'], 'class': g['cls'], 'confidence': p['conf'],
                        'iou': best_iou, **gt_cols(g), **pred_cols(p), 'type': 'localization_disagreement'})
                    flagged = True
                else:
                    self.unmatched_labels.append({
                        'image': img_path.name, 'gt_line': g['line'], 'gt_class': g['cls'], **gt_cols(g),
                        'type': 'unmatched_label'})

            for pj in sorted(free_pred):
                p = preds[pj]
                self.missing_annotations.append({
                    'image': img_path.name, 'pred_class': p['cls'], 'confidence': p['conf'], **pred_cols(p),
                    'type': 'missing_annotation'})
                flagged = True

            if flagged:
                self._draw_and_save(img_path, [g['xyxy'] + [g['cls']] for g in gt],
                                    [p['xyxy'] + [p['cls'], p['conf']] for p in preds], img_path.name)

        report = self._generate_report()
        self._save_csvs()
        logger.info(f"Detection complete. Reports saved to {self.output_dir}")
        return report

    def _generate_report(self) -> Dict:
        return {
            'total_disagreements': len(self.disagreements),
            'total_missing': len(self.missing_annotations),
            'total_localization_issues': len(self.localization_disagreements),
            'total_unmatched_labels': len(self.unmatched_labels),
        }

    def _save_csvs(self):
        outputs = {
            'class_disagreements.csv': self.disagreements,
            'missing_annotations.csv': self.missing_annotations,
            'localization_disagreements.csv': self.localization_disagreements,
            'unmatched_labels.csv': self.unmatched_labels,
        }
        for name, rows in outputs.items():
            path = self.output_dir / name
            if rows:
                pd.DataFrame(rows).to_csv(path, index=False)
            elif path.exists():
                path.unlink()   # don't leave a stale file from an earlier run
