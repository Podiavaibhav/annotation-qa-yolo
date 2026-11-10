"""
Regression tests for pipeline fixes (Issues 1-6).
Uses small synthetic datasets. No GPU or full COCO required.
"""
import pytest
import yaml
import json
import random
from pathlib import Path

from src.dataset_loader import YOLODatasetLoader
from src.analyzer.annotation_analyzer import AnnotationAnalyzer
from src.cleaner.dataset_cleaner import DatasetCleaner
from src.training.noise_injector import NoiseInjector


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a dataset with train and val splits, including known issues."""
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()
    
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True)
    
    # Train image 1: Valid + Exact Duplicate + Malformed
    (dataset_dir / "images" / "train" / "t1.jpg").write_bytes(b"fake")
    (dataset_dir / "labels" / "train" / "t1.txt").write_text(
        "0 0.5 0.5 0.3 0.4\n"       # Valid
        "0 0.5 0.5 0.3 0.4\n"       # Exact duplicate (same class)
        "invalid line\n"            # Malformed format
    )
    
    # Train image 2: Overlapping different class + Tiny box
    (dataset_dir / "images" / "train" / "t2.jpg").write_bytes(b"fake")
    (dataset_dir / "labels" / "train" / "t2.txt").write_text(
        "0 0.5 0.5 0.3 0.4\n"       # Class 0
        "1 0.5 0.5 0.3 0.4\n"       # Overlapping different class (Class 1)
        "0 0.5 0.5 0.0001 0.0001\n" # Tiny box
    )
    
    # Val image: Should remain untouched
    (dataset_dir / "images" / "val" / "v1.jpg").write_bytes(b"fake")
    (dataset_dir / "labels" / "val" / "v1.txt").write_text("0 0.5 0.5 0.3 0.4\n")

    config = {'path': '.', 'train': 'images/train', 'val': 'images/val', 'nc': 2, 'names': ['cat', 'dog']}
    with open(dataset_dir / "data.yaml", 'w') as f:
        yaml.dump(config, f)
    
    return dataset_dir


@pytest.fixture
def single_class_dataset(tmp_path):
    """Dataset with only 1 class to test safe class flipping."""
    dataset_dir = tmp_path / "single_class"
    dataset_dir.mkdir()
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "train" / "img1.jpg").write_bytes(b"fake")
    (dataset_dir / "labels" / "train" / "img1.txt").write_text("0 0.5 0.5 0.3 0.4\n")
    
    config = {'path': '.', 'train': 'images/train', 'nc': 1, 'names': ['cat']}
    with open(dataset_dir / "data.yaml", 'w') as f:
        yaml.dump(config, f)
    return dataset_dir


class TestPipelineFixes:
    def test_duplicate_detection_retains_class_ids(self, synthetic_dataset, tmp_path):
        """Issue 5: Duplicate detection must distinguish same-class from different-class overlaps."""
        config = {'output_dir': str(tmp_path / 'out'), 'duplicate_iou_threshold': 0.98}
        analyzer = AnnotationAnalyzer(str(synthetic_dataset), config)
        report = analyzer.analyze()
        
        issues = [item['issue'] for item in report['suspicious_details']]
        assert 'exact_duplicate' in issues
        assert 'overlapping_different_class' in issues
        
        # Verify class IDs are retained in details
        details = [item['details'] for item in report['suspicious_details'] if item['issue'] == 'exact_duplicate']
        assert any('other_class=0' in d for d in details)

    def test_malformed_labels_reported_with_line_numbers(self, synthetic_dataset, tmp_path):
        """Issue 6: Malformed labels must be reported with consistent line numbers."""
        config = {'output_dir': str(tmp_path / 'out')}
        analyzer = AnnotationAnalyzer(str(synthetic_dataset), config)
        report = analyzer.analyze()
        
        invalid = [item for item in report['invalid_details'] if item['image'] == 't1.txt']
        assert any(item['issue'] == 'invalid_format' and item['line'] == 3 for item in invalid)

    def test_cleaner_preserves_all_splits_and_only_modifies_train(self, synthetic_dataset, tmp_path):
        """Issue 2: Cleaner must preserve val/test splits untouched and only modify train."""
        config = {'output_dir': str(tmp_path / 'out'), 'cleaning_mode': 'conservative'}
        
        # Create a fake analysis report to trigger cleaning
        report_path = tmp_path / 'report.json'
        with open(report_path, 'w') as f:
            json.dump({'invalid_details': [{'image': 't1.txt', 'line': 3, 'issue': 'invalid_format'}]}, f)
            
        cleaner = DatasetCleaner(str(synthetic_dataset), config, analysis_report_path=str(report_path))
        cleaner.clean()
        
        cleaned_dir = cleaner.cleaned_dataset_dir
        
        # Val split must be perfectly preserved
        assert (cleaned_dir / 'images' / 'val' / 'v1.jpg').exists()
        assert (cleaned_dir / 'labels' / 'val' / 'v1.txt').exists()
        with open(cleaned_dir / 'labels' / 'val' / 'v1.txt') as f:
            assert f.read().strip() == "0 0.5 0.5 0.3 0.4"
            
        # Train split must have the malformed line removed
        with open(cleaned_dir / 'labels' / 'train' / 't1.txt') as f:
            lines = f.readlines()
            assert len(lines) == 2  # Original had 3, 1 was malformed and removed

    def test_noise_injector_uses_seed_and_handles_single_class(self, single_class_dataset, tmp_path):
        """Issue 3: Noise injector must use seeded random and handle single-class datasets safely."""
        config = {
            'output_dir': str(tmp_path / 'out'),
            'seed': 42,
            'noise': {
                'delete_probability': 0.0,
                'class_flip_probability': 1.0,  # Force flip attempt
                'coordinate_shift_probability': 0.0,
                'resize_probability': 0.0,
                'duplicate_probability': 0.0
            }
        }
        
        injector = NoiseInjector(str(single_class_dataset), config)
        noisy_dir = injector.inject()
        
        # With 1 class, class flip should gracefully do nothing (or keep same class)
        # and not crash with "empty sequence" error from random.choice
        label_file = noisy_dir / 'labels' / 'train' / 'img1.txt'
        assert label_file.exists()
        with open(label_file) as f:
            content = f.read().strip()
            assert content == "0 0.500000 0.500000 0.300000 0.400000"

    def test_cleaner_loads_model_audit_for_aggressive_mode(self, synthetic_dataset, tmp_path):
        """Issue 4: Cleaner must properly load model audit CSV in aggressive mode."""
        config = {'output_dir': str(tmp_path / 'out'), 'cleaning_mode': 'aggressive'}
        
        # Create fake model audit CSV
        audit_dir = tmp_path / 'model_audit'
        audit_dir.mkdir()
        with open(audit_dir / 'class_disagreements.csv', 'w') as f:
            f.write("image,gt_class,pred_class,confidence,iou\n")
            f.write("t2.txt,0,1,0.95,0.85\n")  # Flags class 0 in t2.txt for removal
            
        cleaner = DatasetCleaner(str(synthetic_dataset), config, model_audit_path=str(audit_dir))
        
        # Verify it loaded the disagreement
        assert ('t2.txt', 0) in cleaner.suspicious_to_remove
