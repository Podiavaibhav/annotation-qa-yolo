"""
Tests for Module 4: Training, Evaluation, and Noise Injection.
Uses mocks to avoid actual model training during unit tests.
"""
import pytest
import yaml
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.training.noise_injector import NoiseInjector
from src.training.trainer import train_model
from src.training.evaluator import evaluate_model
from src.training.report_generator import ReportGenerator


@pytest.fixture
def training_dataset(tmp_path):
    dataset_dir = tmp_path / "training_dataset"
    dataset_dir.mkdir()
    
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True)
    
    (dataset_dir / "images" / "train" / "img1.jpg").write_bytes(b"fake")
    (dataset_dir / "labels" / "train" / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n" * 10)
    
    (dataset_dir / "images" / "val" / "img2.jpg").write_bytes(b"fake")
    (dataset_dir / "labels" / "val" / "img2.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    config = {
        'path': str(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['cat']
    }
    with open(dataset_dir / "data.yaml", 'w') as f:
        yaml.dump(config, f)
    
    return dataset_dir


class TestNoiseInjector:
    def test_injection_creates_noisy_dataset(self, training_dataset, tmp_path):
        config = {
            'output_dir': str(tmp_path / 'outputs'),
            'noise': {
                'delete_probability': 1.0,
                'class_flip_probability': 0.0,
                'coordinate_shift_probability': 0.0,
                'resize_probability': 0.0,
                'duplicate_probability': 0.0
            }
        }
        
        injector = NoiseInjector(str(training_dataset), config)
        noisy_path = injector.inject()
        
        assert noisy_path.exists()
        assert (noisy_path / "corruption_manifest.json").exists()
        
        noisy_label = noisy_path / "labels" / "train" / "img1.txt"
        with open(noisy_label) as f:
            assert len(f.readlines()) == 0


class MockMetricsBox:
    def __init__(self):
        self.map50 = 0.85
        self.map = 0.65
        self.mp = 0.88
        self.mr = 0.82

class MockMetrics:
    def __init__(self):
        self.box = MockMetricsBox()

class MockTrainResults:
    def __init__(self, save_dir):
        self.save_dir = save_dir

class MockYOLO:
    def train(self, **kwargs):
        project = Path(kwargs['project'])
        project.mkdir(parents=True, exist_ok=True)
        (project / "weights").mkdir(exist_ok=True)
        (project / "weights" / "best.pt").touch()
        return MockTrainResults(str(project))
        
    def val(self, **kwargs):
        return MockMetrics()


class TestTrainerAndEvaluator:
    @patch('src.training.trainer.YOLO', return_value=MockYOLO())
    def test_trainer_calls_yolo(self, mock_yolo_class, training_dataset, tmp_path):
        config = {'model_name': 'yolov8n.pt', 'epochs': 1, 'img_size': 640, 'batch_size': 16, 'seed': 42, 'optimizer': 'auto'}
        yaml_path = str(training_dataset / "data.yaml")
        out_dir = str(tmp_path / "train_out")
        
        model_path = train_model(yaml_path, out_dir, config)
        
        mock_yolo_class.assert_called_once_with('yolov8n.pt')
        assert model_path.exists()

    @patch('src.training.evaluator.YOLO', return_value=MockYOLO())
    def test_evaluator_returns_metrics(self, mock_yolo_class, tmp_path):
        (tmp_path / "best.pt").touch()
        metrics = evaluate_model(str(tmp_path / "best.pt"), "dummy.yaml", str(tmp_path / "eval_out"))
        
        assert metrics['mAP50'] == 0.85
        assert metrics['precision'] == 0.88


class TestReportGenerator:
    def test_generates_files(self, tmp_path):
        results = {
            'Original': {'mAP50': 0.85, 'mAP50-95': 0.65, 'precision': 0.88, 'recall': 0.82},
            'Noisy': {'mAP50': 0.65, 'mAP50-95': 0.45, 'precision': 0.70, 'recall': 0.60},
            'Cleaned': {'mAP50': 0.80, 'mAP50-95': 0.60, 'precision': 0.83, 'recall': 0.77}
        }
        
        gen = ReportGenerator(str(tmp_path))
        gen.generate_comparison(results)
        gen.generate_markdown_report(results)
        
        assert (gen.output_dir / 'comparison.csv').exists()
        assert (gen.output_dir / 'comparison.json').exists()
        assert (gen.output_dir / 'performance_comparison.png').exists()
        assert (gen.output_dir / 'final_report.md').exists()
