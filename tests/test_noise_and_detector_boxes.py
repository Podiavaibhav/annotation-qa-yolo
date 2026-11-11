"""
Tests for: noise that is strong enough to matter, noisy yaml pointing at the noisy copy,
and detector flags that carry box coordinates and line numbers.
"""
import json
import yaml
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from src.detector.model_disagreement import ModelDisagreementDetector
from src.training.noise_injector import NoiseInjector

ONLY = {'delete_probability': 0.0, 'class_flip_probability': 0.0, 'coordinate_shift_probability': 0.0,
        'resize_probability': 0.0, 'duplicate_probability': 0.0}


def _write_dataset(root, layout="standard", with_cache=False, download=False):
    if layout == "standard":
        img, lab = root / "images" / "train", root / "labels" / "train"
        vimg, vlab = root / "images" / "val", root / "labels" / "val"
        train_entry, val_entry = "images/train", "images/val"
    else:
        img, lab = root / "train" / "images", root / "train" / "labels"
        vimg, vlab = root / "valid" / "images", root / "valid" / "labels"
        train_entry, val_entry = "train/images", "valid/images"
    for d in (img, lab, vimg, vlab):
        d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(40):
        (img / f"im{i}.jpg").write_bytes(b"x")
        lines = []
        for cx, cy in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]:
            lines.append(f"{int(rng.integers(0, 3))} {cx} {cy} 0.2 0.2")
        (lab / f"im{i}.txt").write_text("\n".join(lines) + "\n")
    (vimg / "v0.jpg").write_bytes(b"x")
    (vlab / "v0.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    cfg = {'path': 'datasets/nowhere', 'train': train_entry, 'val': val_entry, 'names': ['a', 'b', 'c']}
    if download:
        cfg['download'] = 'print("do not run me")'
    with open(root / "data.yaml", 'w') as f:
        yaml.safe_dump(cfg, f)
    if with_cache:
        (lab.parent / "train.cache").write_bytes(b"stale")
    return root


@pytest.fixture
def dataset(tmp_path):
    return _write_dataset(tmp_path / "ds")


def _inject(dataset, tmp_path, **noise):
    config = {'output_dir': str(tmp_path / 'out'), 'seed': 7, 'noise': {**ONLY, **noise}}
    injector = NoiseInjector(str(dataset), config)
    return injector, injector.inject()


class TestNoiseStrength:
    def test_shifts_are_real_errors(self, dataset, tmp_path):
        injector, _ = _inject(dataset, tmp_path, coordinate_shift_probability=1.0)
        shifts = [e for e in injector.manifest if e['type'] == 'coordinate_shift']
        assert len(shifts) == 160
        assert max(e['iou_with_original'] for e in shifts) < 0.7
        assert min(e['iou_with_original'] for e in shifts) > 0.05   # moved, not teleported

    def test_resizes_are_real_errors(self, dataset, tmp_path):
        injector, _ = _inject(dataset, tmp_path, resize_probability=1.0)
        resizes = [e for e in injector.manifest if e['type'] == 'resize']
        assert len(resizes) == 160
        assert max(e['iou_with_original'] for e in resizes) < 0.7

    def test_boxes_stay_inside_the_image(self, tmp_path):
        root = tmp_path / "edge"
        _write_dataset(root)
        for f in (root / "labels" / "train").glob("*.txt"):
            f.write_text("0 0.06 0.94 0.1 0.1\n1 0.96 0.05 0.08 0.08\n")   # hugging the corners
        _, noisy = _inject(root, tmp_path, coordinate_shift_probability=0.5, resize_probability=0.5)
        for f in (noisy / "labels" / "train").glob("*.txt"):
            for line in f.read_text().splitlines():
                _, x, y, w, h = [float(v) for v in line.split()]
                assert x - w / 2 >= -1e-6 and x + w / 2 <= 1 + 1e-6
                assert y - h / 2 >= -1e-6 and y + h / 2 <= 1 + 1e-6

    def test_default_noise_corrupts_a_meaningful_share(self, dataset, tmp_path):
        config = {'output_dir': str(tmp_path / 'out'), 'seed': 7}
        injector = NoiseInjector(str(dataset), config)
        injector.inject()
        share = len({(e['file'], e['line']) for e in injector.manifest}) / 160
        assert 0.15 < share < 0.45


class TestNoisyYaml:
    def test_yaml_points_at_the_noisy_copy_and_drops_download(self, tmp_path):
        ds = _write_dataset(tmp_path / "ds", download=True)
        _, noisy = _inject(ds, tmp_path, delete_probability=1.0)
        cfg = yaml.safe_load((noisy / "data.yaml").read_text())
        assert Path(cfg['path']) == noisy.resolve()
        assert 'download' not in cfg

    def test_stale_label_cache_is_not_copied(self, tmp_path):
        ds = _write_dataset(tmp_path / "ds", with_cache=True)
        _, noisy = _inject(ds, tmp_path, delete_probability=1.0)
        assert not list(noisy.rglob("*.cache"))

    def test_split_folder_layout_is_supported(self, tmp_path):
        ds = _write_dataset(tmp_path / "ds", layout="split")
        injector, noisy = _inject(ds, tmp_path, delete_probability=1.0)
        assert all(e['file'].startswith("train/labels/") for e in injector.manifest)
        assert all(f.read_text() == "" for f in (noisy / "train" / "labels").glob("*.txt"))


# ----------------------------------------------------------------------------- detector
class _T:
    def __init__(self, v):
        self.v = v

    def cpu(self):
        return self

    def numpy(self):
        return self

    def tolist(self):
        return self.v

    def item(self):
        return self.v


class _Col:
    def __init__(self, vals):
        self.vals = vals

    def __getitem__(self, i):
        return _T(self.vals[i])


class _Boxes:
    def __init__(self, preds):
        self.xyxy, self.conf, self.cls = (_Col([p[k] for p in preds]) for k in range(3))
        self._n = len(preds)

    def __len__(self):
        return self._n


class _Result:
    def __init__(self, preds):
        self.boxes = _Boxes(preds)


class FakeYOLO:
    """preds: {image name: [(xyxy, conf, cls), ...]}"""
    def __init__(self, preds):
        self.preds = preds

    def __call__(self, img_path, **kwargs):
        return [_Result(self.preds.get(Path(img_path).name, []))]


def _detector_dataset(tmp_path, labels):
    import cv2
    root = tmp_path / "det"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    for name, text in labels.items():
        cv2.imwrite(str(root / "images" / "train" / f"{name}.jpg"), np.zeros((100, 100, 3), np.uint8))
        (root / "labels" / "train" / f"{name}.txt").write_text(text)
    with open(root / "data.yaml", 'w') as f:
        yaml.safe_dump({'path': str(root), 'train': 'images/train', 'nc': 3, 'names': ['a', 'b', 'c']}, f)
    return root, {'output_dir': str(tmp_path / 'out')}


class TestDetectorBoxes:
    def test_flags_carry_line_numbers_and_coordinates(self, tmp_path):
        root, cfg = _detector_dataset(tmp_path, {'im': "0 0.2 0.2 0.1 0.1\n0 0.5 0.5 0.2 0.2\n"})
        preds = {'im.jpg': [([40.0, 40.0, 60.0, 60.0], 0.9, 1)]}      # line 2: same box, other class
        with patch('src.detector.model_disagreement.YOLO', return_value=FakeYOLO(preds)):
            det = ModelDisagreementDetector(str(root), cfg)
            report = det.detect()
        row = det.disagreements[0]
        assert (row['gt_line'], row['gt_class'], row['pred_class']) == (2, 0, 1)
        assert row['gt_cx'] == pytest.approx(0.5) and row['pred_cx'] == pytest.approx(0.5)
        assert row['pred_w'] == pytest.approx(0.2)
        assert report['total_unmatched_labels'] == 1 and det.unmatched_labels[0]['gt_line'] == 1

    def test_badly_shifted_box_is_one_localization_flag_not_missing_plus_spurious(self, tmp_path):
        root, cfg = _detector_dataset(tmp_path, {'im': "0 0.5 0.5 0.4 0.4\n"})      # box 30..70
        preds = {'im.jpg': [([50.0, 50.0, 90.0, 90.0], 0.9, 0)]}                    # IoU about 0.14
        with patch('src.detector.model_disagreement.YOLO', return_value=FakeYOLO(preds)):
            det = ModelDisagreementDetector(str(root), cfg)
            report = det.detect()
        assert report['total_localization_issues'] == 1
        assert report['total_missing'] == 0 and report['total_unmatched_labels'] == 0
        row = det.localization_disagreements[0]
        assert row['iou'] == pytest.approx(400 / 2800, abs=1e-3)
        assert row['pred_cx'] == pytest.approx(0.7) and row['gt_cx'] == pytest.approx(0.5)

    def test_each_image_uses_its_own_out_of_fold_model(self, tmp_path):
        root, cfg = _detector_dataset(tmp_path, {'a': "0 0.5 0.5 0.2 0.2\n", 'b': "0 0.5 0.5 0.2 0.2\n"})
        models = {
            'weights_A': FakeYOLO({'a.jpg': [([40.0, 40.0, 60.0, 60.0], 0.9, 2)]}),   # says class 2 for image a
            'weights_B': FakeYOLO({'b.jpg': [([40.0, 40.0, 60.0, 60.0], 0.9, 0)]}),   # agrees for image b
        }
        loaded = []

        def factory(weights):
            loaded.append(weights)
            return models[weights]

        with patch('src.detector.model_disagreement.YOLO', side_effect=factory):
            det = ModelDisagreementDetector(str(root), cfg, fold_models={'a.jpg': 'weights_A', 'b.jpg': 'weights_B'})
            det.detect()
        assert sorted(loaded) == ['weights_A', 'weights_B']          # each weight file loaded once
        assert [d['image'] for d in det.disagreements] == ['a.jpg']  # b agreed, a did not

    def test_missing_out_of_fold_model_is_an_error_not_a_silent_fallback(self, tmp_path):
        root, cfg = _detector_dataset(tmp_path, {'a': "0 0.5 0.5 0.2 0.2\n"})
        with patch('src.detector.model_disagreement.YOLO', return_value=FakeYOLO({})):
            det = ModelDisagreementDetector(str(root), cfg, fold_models={'other.jpg': 'w'})
            with pytest.raises(KeyError):
                det.detect()

    def test_rerun_removes_stale_csvs(self, tmp_path):
        root, cfg = _detector_dataset(tmp_path, {'im': "0 0.5 0.5 0.2 0.2\n"})
        out = Path(cfg['output_dir']) / 'model_audit'
        with patch('src.detector.model_disagreement.YOLO',
                   return_value=FakeYOLO({'im.jpg': [([40.0, 40.0, 60.0, 60.0], 0.9, 1)]})):
            ModelDisagreementDetector(str(root), cfg).detect()
        assert (out / 'class_disagreements.csv').exists()
        with patch('src.detector.model_disagreement.YOLO', return_value=FakeYOLO({})):
            ModelDisagreementDetector(str(root), cfg).detect()
        assert not (out / 'class_disagreements.csv').exists()
