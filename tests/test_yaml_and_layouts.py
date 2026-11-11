"""
Tests for the yaml path fix (the cause of the Kaggle "images not found" error)
and for the dataset layouts the loader understands.
"""
import yaml
from pathlib import Path

from src.dataset_loader import YOLODatasetLoader
from src.yaml_utils import absolutize_yaml, set_yaml_root, write_prepared_yaml


def _dump(path, cfg):
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)


def _make(root, images_rel, labels_rel, n=2):
    (root / images_rel).mkdir(parents=True)
    (root / labels_rel).mkdir(parents=True)
    for i in range(n):
        (root / images_rel / f"im{i}.jpg").write_bytes(b"x")
        (root / labels_rel / f"im{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")


class TestAbsolutizeYaml:
    def test_relative_path_that_points_nowhere_falls_back_to_yaml_folder(self, tmp_path):
        """coco8.yaml says `path: datasets/coco8`; the data is next to the yaml, not there."""
        ds = tmp_path / "ds"
        _make(ds, "images/train", "labels/train")
        _dump(ds / "data.yaml", {'path': 'datasets/coco8', 'train': 'images/train', 'names': ['a'],
                                 'download': 'print("would run")'})
        out = absolutize_yaml(ds / "data.yaml", tmp_path / "out" / "fixed.yaml")
        cfg = yaml.safe_load(out.read_text())
        assert Path(cfg['path']) == ds.resolve()
        assert 'download' not in cfg
        assert cfg['train'] == 'images/train'

    def test_dot_path_means_yaml_folder(self, tmp_path):
        ds = tmp_path / "ds"
        _make(ds, "images/train", "labels/train")
        _dump(ds / "data.yaml", {'path': '.', 'train': 'images/train', 'names': ['a']})
        cfg = yaml.safe_load(absolutize_yaml(ds / "data.yaml", tmp_path / "f.yaml").read_text())
        assert Path(cfg['path']) == ds.resolve()

    def test_existing_absolute_path_elsewhere_is_kept(self, tmp_path):
        data = tmp_path / "data"
        _make(data, "images/train", "labels/train")
        configs = tmp_path / "configs"
        configs.mkdir()
        _dump(configs / "x.yaml", {'path': str(data), 'train': 'images/train', 'names': ['a']})
        cfg = yaml.safe_load(absolutize_yaml(configs / "x.yaml", tmp_path / "f.yaml").read_text())
        assert Path(cfg['path']) == data

    def test_explicit_dataset_root_wins(self, tmp_path):
        ds = tmp_path / "ds"
        ds.mkdir()
        _dump(ds / "data.yaml", {'path': 'whatever', 'train': 'images/train', 'names': ['a']})
        other = tmp_path / "other"
        other.mkdir()
        cfg = yaml.safe_load(absolutize_yaml(ds / "data.yaml", tmp_path / "f.yaml", dataset_root=other).read_text())
        assert Path(cfg['path']) == other.resolve()

    def test_set_yaml_root_rewrites_in_place(self, tmp_path):
        y = tmp_path / "d.yaml"
        _dump(y, {'path': 'relative/thing', 'names': ['a'], 'download': 'x'})
        set_yaml_root(y, tmp_path)
        cfg = yaml.safe_load(y.read_text())
        assert Path(cfg['path']) == tmp_path.resolve() and 'download' not in cfg


class TestPreparedYaml:
    def test_write_prepared_yaml_uses_relative_splits_and_absolute_path(self, tmp_path):
        root = tmp_path / "african-wildlife"
        info = {'path': root, 'train': root / 'train' / 'images', 'val': root / 'valid' / 'images',
                'names': {0: 'buffalo', 1: 'elephant'}}
        root.mkdir()
        out = write_prepared_yaml(info)
        cfg = yaml.safe_load(out.read_text())
        assert out == root.resolve() / "data.yaml"
        assert Path(cfg['path']) == root.resolve()
        assert cfg['train'] == str(Path('train') / 'images') and cfg['val'] == str(Path('valid') / 'images')
        assert cfg['nc'] == 2 and cfg['names'] == {0: 'buffalo', 1: 'elephant'}


class TestLoaderLayouts:
    def test_split_folder_layout_with_valid_alias(self, tmp_path):
        """Layout used by several Ultralytics datasets: <split>/images, <split>/labels, 'valid' not 'val'."""
        ds = tmp_path / "wild"
        _make(ds, "train/images", "train/labels", n=3)
        _make(ds, "valid/images", "valid/labels", n=2)
        _dump(ds / "data.yaml", {'path': str(ds), 'train': 'train/images', 'val': 'valid/images',
                                 'names': {0: 'buffalo', 1: 'zebra'}})
        loader = YOLODatasetLoader(str(ds))
        assert len(loader.get_image_label_pairs('train')) == 3
        assert len(loader.get_image_label_pairs('val')) == 2
        assert loader.num_classes == 2
        assert loader.validate_dataset() == []

    def test_yaml_entry_decides_when_folders_are_unusual(self, tmp_path):
        ds = tmp_path / "odd"
        _make(ds, "pics/images/tr", "pics/labels/tr")
        _dump(ds / "data.yaml", {'path': '.', 'train': 'pics/images/tr', 'names': ['a']})
        loader = YOLODatasetLoader(str(ds))
        images, labels = loader.get_split_paths('train')
        assert images == ds / "pics" / "images" / "tr" and labels == ds / "pics" / "labels" / "tr"

    def test_relative_path_pointing_nowhere_is_ignored(self, tmp_path):
        ds = tmp_path / "coco8like"
        _make(ds, "images/train", "labels/train")
        _dump(ds / "coco8.yaml", {'path': 'datasets/coco8', 'train': 'images/train', 'names': ['a']})
        assert len(YOLODatasetLoader(str(ds / "coco8.yaml")).get_image_label_pairs('train')) == 2

    def test_missing_split_raises(self, tmp_path):
        ds = tmp_path / "none"
        ds.mkdir()
        _dump(ds / "data.yaml", {'path': '.', 'train': 'images/train', 'names': ['a']})
        try:
            YOLODatasetLoader(str(ds)).get_split_paths('train')
        except FileNotFoundError as e:
            assert 'train' in str(e)
        else:
            raise AssertionError("expected FileNotFoundError")
