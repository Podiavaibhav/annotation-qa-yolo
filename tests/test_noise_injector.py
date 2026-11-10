"""
Tests for Noise Injector.
"""
import pytest
import yaml
import json
from pathlib import Path

from src.training.noise_injector import NoiseInjector


@pytest.fixture
def base_dataset(tmp_path):
    """Create a standard 2-class dataset for testing."""
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()
    
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)
    
    (dataset_dir / "images" / "train" / "img1.jpg").write_bytes(b"fake")
    (dataset_dir / "labels" / "train" / "img1.txt").write_text(
        "0 0.5 0.5 0.3 0.4\n"
        "1 0.2 0.3 0.1 0.2\n"
    )
    
    config = {'path': '.', 'train': 'images/train', 'nc': 2, 'names': ['cat', 'dog']}
    with open(dataset_dir / "data.yaml", 'w') as f:
        yaml.dump(config, f)
    
    return dataset_dir


@pytest.fixture
def single_class_dataset(tmp_path):
    """Create a 1-class dataset."""
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


class TestNoiseInjector:
    def test_reproducibility_with_identical_seeds(self, base_dataset, tmp_path):
        config1 = {'output_dir': str(tmp_path / 'out1'), 'seed': 42, 'noise': {'delete_probability': 1.0, 'class_flip_probability': 0.0, 'coordinate_shift_probability': 0.0, 'resize_probability': 0.0, 'duplicate_probability': 0.0}}
        config2 = {'output_dir': str(tmp_path / 'out2'), 'seed': 42, 'noise': {'delete_probability': 1.0, 'class_flip_probability': 0.0, 'coordinate_shift_probability': 0.0, 'resize_probability': 0.0, 'duplicate_probability': 0.0}}
        
        injector1 = NoiseInjector(str(base_dataset), config1)
        injector1.inject()
        
        injector2 = NoiseInjector(str(base_dataset), config2)
        injector2.inject()
        
        with open(injector1.output_dir / "corruption_manifest.json") as f1, \
             open(injector2.output_dir / "corruption_manifest.json") as f2:
            assert json.load(f1) == json.load(f2)

    def test_probability_validation(self, base_dataset, tmp_path):
        with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
            NoiseInjector(str(base_dataset), {'noise': {'delete_probability': -0.1}})
            
        with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
            NoiseInjector(str(base_dataset), {'noise': {'delete_probability': 1.5}})
            
        with pytest.raises(ValueError, match="cannot exceed 1.0"):
            NoiseInjector(str(base_dataset), {'noise': {
                'delete_probability': 0.5, 'class_flip_probability': 0.5,
                'coordinate_shift_probability': 0.5, 'resize_probability': 0.5
            }})

    def test_single_class_dataset_safe_handling(self, single_class_dataset, tmp_path):
        config = {
            'output_dir': str(tmp_path / 'out'), 'seed': 42,
            'noise': {'delete_probability': 0.0, 'class_flip_probability': 1.0, 'coordinate_shift_probability': 0.0, 'resize_probability': 0.0, 'duplicate_probability': 0.0}
        }
        
        injector = NoiseInjector(str(single_class_dataset), config)
        injector.inject()
        
        # Should not crash, and should not record a class flip
        assert len(injector.manifest) == 0
        
        # Original annotation should remain unchanged
        label_file = injector.output_dir / "labels" / "train" / "img1.txt"
        with open(label_file) as f:
            assert f.read().strip() == "0 0.5 0.5 0.3 0.4"

    def test_corruption_manifest_schema(self, base_dataset, tmp_path):
        config = {
            'output_dir': str(tmp_path / 'out'), 'seed': 42,
            'noise': {'delete_probability': 0.25, 'class_flip_probability': 0.25, 'coordinate_shift_probability': 0.25, 'resize_probability': 0.25, 'duplicate_probability': 0.5}
        }
        injector = NoiseInjector(str(base_dataset), config)
        injector.inject()
        
        for entry in injector.manifest:
            assert 'file' in entry
            assert 'line' in entry
            assert 'type' in entry
            assert 'original' in entry
            assert 'result' in entry or entry['type'] == 'deleted'
            
            orig = entry['original']
            assert all(k in orig for k in ['cls', 'x', 'y', 'w', 'h'])
            
            if entry['type'] != 'deleted':
                res = entry['result']
                assert all(k in res for k in ['cls', 'x', 'y', 'w', 'h'])

    def test_duplicate_injection(self, base_dataset, tmp_path):
        config = {
            'output_dir': str(tmp_path / 'out'), 'seed': 42,
            'noise': {'delete_probability': 0.0, 'class_flip_probability': 0.0, 'coordinate_shift_probability': 0.0, 'resize_probability': 0.0, 'duplicate_probability': 1.0}
        }
        injector = NoiseInjector(str(base_dataset), config)
        injector.inject()
        
        dup_entries = [e for e in injector.manifest if e['type'] == 'duplicate']
        assert len(dup_entries) == 2  # 2 lines in original file
        
        label_file = injector.output_dir / "labels" / "train" / "img1.txt"
        with open(label_file) as f:
            lines = f.readlines()
        assert len(lines) == 4  # 2 original + 2 duplicates

    def test_correct_relative_manifest_paths(self, base_dataset, tmp_path):
        config = {
            'output_dir': str(tmp_path / 'out'), 'seed': 42,
            'noise': {'delete_probability': 1.0, 'class_flip_probability': 0.0, 'coordinate_shift_probability': 0.0, 'resize_probability': 0.0, 'duplicate_probability': 0.0}
        }
        injector = NoiseInjector(str(base_dataset), config)
        injector.inject()
        
        for entry in injector.manifest:
            assert entry['file'] == "labels/train/img1.txt"

    def test_original_dataset_unchanged(self, base_dataset, tmp_path):
        config = {
            'output_dir': str(tmp_path / 'out'), 'seed': 42,
            'noise': {'delete_probability': 1.0, 'class_flip_probability': 0.0, 'coordinate_shift_probability': 0.0, 'resize_probability': 0.0, 'duplicate_probability': 0.0}
        }
        
        orig_label = base_dataset / "labels" / "train" / "img1.txt"
        with open(orig_label) as f:
            orig_content = f.read()
            
        injector = NoiseInjector(str(base_dataset), config)
        injector.inject()
        
        with open(orig_label) as f:
            assert f.read() == orig_content
