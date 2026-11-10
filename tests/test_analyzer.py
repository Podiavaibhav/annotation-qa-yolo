"""
Tests for Module 1: Annotation Analyzer.
"""
import pytest
import yaml
from pathlib import Path

from src.analyzer.annotation_analyzer import AnnotationAnalyzer


@pytest.fixture
def analyzer_dataset(tmp_path):
    """Create a dataset with specific known issues for testing."""
    dataset_dir = tmp_path / "analyzer_dataset"
    dataset_dir.mkdir()
    
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)
    
    # Image 1: Valid, but has a tiny box and a duplicate
    (dataset_dir / "images" / "train" / "img1.jpg").write_bytes(b"fake")
    (dataset_dir / "labels" / "train" / "img1.txt").write_text(
        "0 0.5 0.5 0.3 0.4\n"       # Valid
        "0 0.5 0.5 0.001 0.001\n"   # Tiny box
        "0 0.5 0.5 0.3 0.4\n"       # Exact duplicate
    )
    
    # Image 2: Invalid format, negative width, invalid class
    (dataset_dir / "images" / "train" / "img2.jpg").write_bytes(b"fake")
    (dataset_dir / "labels" / "train" / "img2.txt").write_text(
        "0 0.5 0.5 0.3 0.4\n"       # Valid
        "invalid line\n"            # Malformed
        "0 0.5 0.5 -0.1 0.4\n"     # Negative width
        "5 0.5 0.5 0.3 0.4\n"      # Invalid class (only 2 classes)
    )
    
    # Image 3: Extreme aspect ratio
    (dataset_dir / "images" / "train" / "img3.jpg").write_bytes(b"fake")
    (dataset_dir / "labels" / "train" / "img3.txt").write_text(
        "1 0.5 0.5 0.9 0.01\n"      # Extreme aspect ratio
    )

    config = {
        'path': str(dataset_dir),
        'train': 'images/train',
        'nc': 2,
        'names': ['cat', 'dog']
    }
    with open(dataset_dir / "data.yaml", 'w') as f:
        yaml.dump(config, f)
    
    return dataset_dir


@pytest.fixture
def default_config():
    return {
        'output_dir': 'outputs',
        'tiny_box_threshold': 0.001,
        'duplicate_iou_threshold': 0.98,
        'max_box_area': 0.95,
        'min_aspect_ratio': 0.1,
        'max_aspect_ratio': 10.0
    }


class TestAnnotationAnalyzer:
    def test_invalid_format_detection(self, analyzer_dataset, default_config):
        analyzer = AnnotationAnalyzer(str(analyzer_dataset), default_config)
        report = analyzer.analyze()
        
        invalid_issues = [i['issue'] for i in report['invalid_details']]
        assert 'invalid_format' in invalid_issues

    def test_invalid_dimensions_detection(self, analyzer_dataset, default_config):
        analyzer = AnnotationAnalyzer(str(analyzer_dataset), default_config)
        report = analyzer.analyze()
        
        invalid_issues = [i['issue'] for i in report['invalid_details']]
        assert 'zero_or_negative_dim' in invalid_issues

    def test_invalid_class_id_detection(self, analyzer_dataset, default_config):
        analyzer = AnnotationAnalyzer(str(analyzer_dataset), default_config)
        report = analyzer.analyze()
        
        invalid_issues = [i['issue'] for i in report['invalid_details']]
        assert 'invalid_class_id' in invalid_issues

    def test_tiny_box_detection(self, analyzer_dataset, default_config):
        analyzer = AnnotationAnalyzer(str(analyzer_dataset), default_config)
        report = analyzer.analyze()
        
        suspicious_issues = [i['issue'] for i in report['suspicious_details']]
        assert 'tiny_box' in suspicious_issues

    def test_duplicate_box_detection(self, analyzer_dataset, default_config):
        analyzer = AnnotationAnalyzer(str(analyzer_dataset), default_config)
        report = analyzer.analyze()
        
        suspicious_issues = [i['issue'] for i in report['suspicious_details']]
        assert 'duplicate_box' in suspicious_issues

    def test_extreme_aspect_ratio_detection(self, analyzer_dataset, default_config):
        analyzer = AnnotationAnalyzer(str(analyzer_dataset), default_config)
        report = analyzer.analyze()
        
        suspicious_issues = [i['issue'] for i in report['suspicious_details']]
        assert 'extreme_aspect_ratio' in suspicious_issues

    def test_output_files_created(self, analyzer_dataset, default_config):
        analyzer = AnnotationAnalyzer(str(analyzer_dataset), default_config)
        analyzer.analyze()
        
        output_dir = Path(default_config['output_dir']) / 'analysis'
        assert (output_dir / 'class_distribution.png').exists()
        assert (output_dir / 'box_area_distribution.png').exists()
        assert (output_dir / 'suspicious_annotations.csv').exists()
