"""
Tests for Module 3: Dataset Cleaning Engine.
"""
import pytest
import yaml
import json
from pathlib import Path

from src.cleaner.dataset_cleaner import DatasetCleaner


@pytest.fixture
def cleaning_dataset(tmp_path):
    """Create a dataset with known invalid and suspicious boxes."""
    dataset_dir = tmp_path / "cleaning_dataset"
    dataset_dir.mkdir()
    
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)
    
    # img1: One valid, one invalid (negative width)
    (dataset_dir / "images" / "train" / "img1.jpg").write_bytes(b"fake")
    (dataset_dir / "labels" / "train" / "img1.txt").write_text(
        "0 0.5 0.5 0.3 0.4\n"       # Valid
        "0 0.5 0.5 -0.1 0.4\n"      # Invalid
    )
    
    # img2: One valid, one suspicious (tiny box)
    (dataset_dir / "images" / "train" / "img2.jpg").write_bytes(b"fake")
    (dataset_dir / "labels" / "train" / "img2.txt").write_text(
        "1 0.5 0.5 0.3 0.4\n"       # Valid
        "1 0.5 0.5 0.0001 0.0001\n" # Suspicious (tiny)
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
def mock_reports(tmp_path, cleaning_dataset):
    """Create mock reports for the cleaner to consume."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    
    # Analysis report
    analysis_report = {
        'invalid_details': [
            {'image': 'img1.jpg', 'line': 2, 'issue': 'zero_or_negative_dim'}
        ],
        'suspicious_details': [
            {'image': 'img2.jpg', 'box': [0.5, 0.5, 0.0001, 0.0001], 'issue': 'tiny_box'}
        ]
    }
    with open(report_dir / 'analysis.json', 'w') as f:
        json.dump(analysis_report, f)
        
    return report_dir / 'analysis.json'


class TestDatasetCleaner:
    def test_conservative_mode(self, cleaning_dataset, mock_reports, tmp_path):
        config = {
            'output_dir': str(tmp_path / 'outputs'),
            'cleaning_mode': 'conservative'
        }
        
        cleaner = DatasetCleaner(str(cleaning_dataset), config, analysis_report_path=str(mock_reports))
        result = cleaner.clean()
        
        # Conservative should only remove the invalid box in img1
        assert result['removed'] == 1
        assert result['kept'] == 3
        
        # Check img2 still has 2 boxes (tiny box is kept in conservative)
        cleaned_label = Path(config['output_dir']) / 'cleaned_dataset' / 'labels' / 'train' / 'img2.txt'
        with open(cleaned_label) as f:
            assert len(f.readlines()) == 2

    def test_aggressive_mode(self, cleaning_dataset, mock_reports, tmp_path):
        config = {
            'output_dir': str(tmp_path / 'outputs'),
            'cleaning_mode': 'aggressive'
        }
        
        cleaner = DatasetCleaner(str(cleaning_dataset), config, analysis_report_path=str(mock_reports))
        result = cleaner.clean()
        
        # Aggressive should remove invalid (1) and suspicious (1)
        assert result['removed'] == 2
        assert result['kept'] == 2
        
        # Check img2 only has 1 box now
        cleaned_label = Path(config['output_dir']) / 'cleaned_dataset' / 'labels' / 'train' / 'img2.txt'
        with open(cleaned_label) as f:
            assert len(f.readlines()) == 1

    def test_original_dataset_untouched(self, cleaning_dataset, mock_reports, tmp_path):
        config = {
            'output_dir': str(tmp_path / 'outputs'),
            'cleaning_mode': 'aggressive'
        }
        
        # Read original line count
        orig_label = cleaning_dataset / "labels" / "train" / "img2.txt"
        with open(orig_label) as f:
            orig_lines = len(f.readlines())
            
        cleaner = DatasetCleaner(str(cleaning_dataset), config, analysis_report_path=str(mock_reports))
        cleaner.clean()
        
        # Verify original is unchanged
        with open(orig_label) as f:
            assert len(f.readlines()) == orig_lines

    def test_yaml_and_images_copied(self, cleaning_dataset, mock_reports, tmp_path):
        config = {
            'output_dir': str(tmp_path / 'outputs'),
            'cleaning_mode': 'conservative'
        }
        
        cleaner = DatasetCleaner(str(cleaning_dataset), config, analysis_report_path=str(mock_reports))
        cleaner.clean()
        
        cleaned_dir = Path(config['output_dir']) / 'cleaned_dataset'
        assert (cleaned_dir / 'data.yaml').exists()
        assert (cleaned_dir / 'images' / 'train' / 'img1.jpg').exists()
        assert (cleaned_dir / 'images' / 'train' / 'img2.jpg').exists()
