"""
Tests for YOLODatasetLoader.
Uses synthetic test fixtures to avoid requiring real datasets.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import yaml

from src.dataset_loader import YOLODatasetLoader


@pytest.fixture
def synthetic_dataset(tmp_path):
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()
    
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True)
    
    (dataset_dir / "images" / "train" / "img1.jpg").write_bytes(b"fake image")
    (dataset_dir / "images" / "train" / "img2.jpg").write_bytes(b"fake image")
    (dataset_dir / "images" / "val" / "img3.jpg").write_bytes(b"fake image")
    
    (dataset_dir / "labels" / "train" / "img1.txt").write_text(
        "0 0.5 0.5 0.3 0.4\n1 0.2 0.3 0.1 0.2\n"
    )
    (dataset_dir / "labels" / "train" / "img2.txt").write_text(
        "0 0.6 0.7 0.2 0.3\n"
    )
    (dataset_dir / "labels" / "val" / "img3.txt").write_text(
        "1 0.4 0.5 0.25 0.35\n"
    )
    
    config = {
        'path': str(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,
        'names': ['cat', 'dog']
    }
    
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    
    return dataset_dir


@pytest.fixture
def malformed_dataset(tmp_path):
    dataset_dir = tmp_path / "malformed_dataset"
    dataset_dir.mkdir()
    
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)
    
    (dataset_dir / "images" / "train" / "img1.jpg").write_bytes(b"fake")
    (dataset_dir / "labels" / "train" / "img1.txt").write_text(
        "0 0.5 0.5 0.3 0.4\n"
        "invalid line\n"
        "1 0.2 0.3\n"
        "2 0.1 0.2 0.3 0.4\n"
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


class TestYOLODatasetLoader:
    def test_load_valid_dataset(self, synthetic_dataset):
        loader = YOLODatasetLoader(str(synthetic_dataset))
        assert loader.num_classes == 2
        assert loader.classes == ['cat', 'dog']
    
    def test_load_from_yaml(self, synthetic_dataset):
        yaml_path = synthetic_dataset / "data.yaml"
        loader = YOLODatasetLoader(str(yaml_path))
        assert loader.num_classes == 2
    
    def test_get_image_label_pairs(self, synthetic_dataset):
        loader = YOLODatasetLoader(str(synthetic_dataset))
        train_pairs = loader.get_image_label_pairs('train')
        val_pairs = loader.get_image_label_pairs('val')
        
        assert len(train_pairs) == 2
        assert len(val_pairs) == 1
        
        for img_path, label_path in train_pairs:
            assert img_path.stem == label_path.stem
    
    def test_load_annotations(self, synthetic_dataset):
        loader = YOLODatasetLoader(str(synthetic_dataset))
        train_pairs = loader.get_image_label_pairs('train')
        
        # Explicitly find img1 to avoid random directory ordering issues
        _, label_path = next(p for p in train_pairs if p[0].stem == 'img1')
        
        annotations = loader.load_annotations(label_path)
        
        assert len(annotations) == 2
        assert annotations[0]['class_id'] == 0
        assert annotations[0]['x_center'] == 0.5
        assert annotations[0]['width'] == 0.3
    
    def test_load_malformed_annotations(self, malformed_dataset):
        loader = YOLODatasetLoader(str(malformed_dataset))
        train_pairs = loader.get_image_label_pairs('train')
        _, label_path = train_pairs[0]
        
        annotations = loader.load_annotations(label_path)
        assert len(annotations) == 2
        assert annotations[0]['class_id'] == 0
        assert annotations[1]['class_id'] == 2
    
    def test_get_dataset_stats(self, synthetic_dataset):
        loader = YOLODatasetLoader(str(synthetic_dataset))
        stats = loader.get_dataset_stats()
        
        assert stats['num_classes'] == 2
        assert stats['splits']['train']['num_images'] == 2
        assert stats['splits']['train']['num_annotations'] == 3
        assert stats['splits']['val']['num_images'] == 1
    
    def test_validate_valid_dataset(self, synthetic_dataset):
        loader = YOLODatasetLoader(str(synthetic_dataset))
        errors = loader.validate_dataset()
        assert len(errors) == 0
    
    def test_validate_missing_split(self, tmp_path):
        dataset_dir = tmp_path / "incomplete_dataset"
        dataset_dir.mkdir()
        
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)
        
        config = {
            'path': str(dataset_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['cat']
        }
        
        with open(dataset_dir / "data.yaml", 'w') as f:
            yaml.dump(config, f)
        
        loader = YOLODatasetLoader(str(dataset_dir))
        errors = loader.validate_dataset()
        
        assert len(errors) > 0
        assert any('val' in error for error in errors)
    
    def test_nonexistent_dataset(self):
        with pytest.raises(FileNotFoundError):
            YOLODatasetLoader("/path/to/nonexistent/dataset")
    
    def test_empty_label_file(self, tmp_path):
        dataset_dir = tmp_path / "empty_labels"
        dataset_dir.mkdir()
        
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)
        
        (dataset_dir / "images" / "train" / "img1.jpg").write_bytes(b"fake")
        (dataset_dir / "labels" / "train" / "img1.txt").write_text("")
        
        config = {
            'path': str(dataset_dir),
            'train': 'images/train',
            'nc': 1,
            'names': ['cat']
        }
        
        with open(dataset_dir / "data.yaml", 'w') as f:
            yaml.dump(config, f)
        
        loader = YOLODatasetLoader(str(dataset_dir))
        train_pairs = loader.get_image_label_pairs('train')
        
        assert len(train_pairs) == 1
        
        _, label_path = train_pairs[0]
        annotations = loader.load_annotations(label_path)
        
        assert len(annotations) == 0
