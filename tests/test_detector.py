"""
Tests for Module 2: Model-Assisted Annotation Error Detection.
Uses mocks to simulate YOLO model predictions.
"""
import pytest
import yaml
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.detector.model_disagreement import ModelDisagreementDetector


class MockTensor:
    """Simulates a PyTorch tensor with .cpu().numpy().tolist() chain."""
    def __init__(self, value):
        self.value = value
    def cpu(self):
        return self
    def numpy(self):
        return np.array(self.value)
    def tolist(self):
        return self.value if isinstance(self.value, list) else [self.value]
    def item(self):
        return self.value


class MockBoxes:
    def __init__(self, predictions):
        self.predictions = predictions
        self._len = len(predictions)
        
    def __len__(self):
        return self._len
        
    @property
    def xyxy(self):
        class XYXYAccessor:
            def __init__(self, predictions):
                self.predictions = predictions
            def __getitem__(self, idx):
                return MockTensor(self.predictions[idx][0])
        return XYXYAccessor(self.predictions)
        
    @property
    def conf(self):
        class ConfAccessor:
            def __init__(self, predictions):
                self.predictions = predictions
            def __getitem__(self, idx):
                return MockTensor(self.predictions[idx][1])
        return ConfAccessor(self.predictions)
        
    @property
    def cls(self):
        class ClsAccessor:
            def __init__(self, predictions):
                self.predictions = predictions
            def __getitem__(self, idx):
                return MockTensor(self.predictions[idx][2])
        return ClsAccessor(self.predictions)


class MockResult:
    def __init__(self, predictions):
        self.boxes = MockBoxes(predictions)
        self.orig_shape = (100, 100)


class MockYOLO:
    def __init__(self, predictions_map):
        self.predictions_map = predictions_map
        
    def __call__(self, img_path, **kwargs):
        img_name = Path(img_path).name
        preds = self.predictions_map.get(img_name, [])
        return [MockResult(preds)]


@pytest.fixture
def detector_dataset(tmp_path):
    dataset_dir = tmp_path / "detector_dataset"
    dataset_dir.mkdir()
    
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)
    
    import cv2
    for i in range(1, 4):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_dir / "images" / "train" / f"img{i}.jpg"), img)
    
    (dataset_dir / "labels" / "train" / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (dataset_dir / "labels" / "train" / "img2.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (dataset_dir / "labels" / "train" / "img3.txt").write_text("")

    config = {
        'path': str(dataset_dir),
        'train': 'images/train',
        'nc': 2,
        'names': ['cat', 'dog'],
        'output_dir': str(tmp_path / 'outputs'),
        'confidence_threshold': 0.5,
        'iou_threshold': 0.5
    }
    with open(dataset_dir / "data.yaml", 'w') as f:
        yaml.dump(config, f)
    
    return dataset_dir, config


class TestModelDisagreementDetector:
    def test_perfect_match_no_flags(self, detector_dataset):
        ds_dir, config = detector_dataset
        
        preds = {
            'img1.jpg': [([40.0, 40.0, 60.0, 60.0], 0.9, 0)],
            'img2.jpg': [],
            'img3.jpg': []
        }
        
        with patch('src.detector.model_disagreement.YOLO', return_value=MockYOLO(preds)):
            detector = ModelDisagreementDetector(str(ds_dir), config)
            report = detector.detect()
            
            assert report['total_disagreements'] == 0
            assert report['total_missing'] == 0

    def test_class_disagreement(self, detector_dataset):
        ds_dir, config = detector_dataset
        
        preds = {
            'img1.jpg': [],
            'img2.jpg': [([40.0, 40.0, 60.0, 60.0], 0.9, 1)],
            'img3.jpg': []
        }
        
        with patch('src.detector.model_disagreement.YOLO', return_value=MockYOLO(preds)):
            detector = ModelDisagreementDetector(str(ds_dir), config)
            report = detector.detect()
            
            assert report['total_disagreements'] == 1
            assert detector.disagreements[0]['gt_class'] == 0
            assert detector.disagreements[0]['pred_class'] == 1

    def test_missing_annotation(self, detector_dataset):
        ds_dir, config = detector_dataset
        
        preds = {
            'img1.jpg': [],
            'img2.jpg': [],
            'img3.jpg': [([40.0, 40.0, 60.0, 60.0], 0.9, 0)]
        }
        
        with patch('src.detector.model_disagreement.YOLO', return_value=MockYOLO(preds)):
            detector = ModelDisagreementDetector(str(ds_dir), config)
            report = detector.detect()
            
            assert report['total_missing'] == 1
            
    def test_csv_generation(self, detector_dataset):
        ds_dir, config = detector_dataset
        
        preds = {
            'img1.jpg': [],
            'img2.jpg': [([40.0, 40.0, 60.0, 60.0], 0.9, 1)],
            'img3.jpg': [([40.0, 40.0, 60.0, 60.0], 0.9, 0)]
        }
        
        with patch('src.detector.model_disagreement.YOLO', return_value=MockYOLO(preds)):
            detector = ModelDisagreementDetector(str(ds_dir), config)
            detector.detect()
            
            output_dir = Path(config['output_dir']) / 'model_audit'
            assert (output_dir / 'class_disagreements.csv').exists()
            assert (output_dir / 'missing_annotations.csv').exists()
