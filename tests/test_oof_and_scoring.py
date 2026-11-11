"""
Tests for out-of-fold model training and for scoring the detector against injected noise.
"""
import json
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch

from src.detector.oof_models import make_folds, train_oof_models
from src.detector.scoring import score_detector


# ----------------------------------------------------------------------------- folds
class TestFolds:
    def test_folds_partition_the_images(self):
        names = [f"im{i}.jpg" for i in range(11)]
        folds = make_folds(names, 3, seed=1)
        flat = [n for f in folds for n in f]
        assert sorted(flat) == sorted(names) and len(flat) == len(set(flat))
        assert max(map(len, folds)) - min(map(len, folds)) <= 1

    def test_same_seed_same_folds_different_seed_different(self):
        names = [f"im{i}.jpg" for i in range(20)]
        assert make_folds(names, 2, 5) == make_folds(list(reversed(names)), 2, 5)   # order of input is irrelevant
        assert make_folds(names, 2, 5) != make_folds(names, 2, 6)

    def test_bad_arguments(self):
        with pytest.raises(ValueError):
            make_folds(["a.jpg", "b.jpg"], 1, 0)
        with pytest.raises(ValueError):
            make_folds(["a.jpg"], 2, 0)


class FakeTrainer:
    """Stands in for ultralytics.YOLO; records every training call."""
    calls = []

    def __init__(self, *a, **k):
        pass

    def train(self, **kw):
        FakeTrainer.calls.append(kw)
        save_dir = Path(kw['project']) / kw['name']
        (save_dir / 'weights').mkdir(parents=True, exist_ok=True)
        (save_dir / 'weights' / 'last.pt').write_text("w")

        class R:
            pass
        r = R()
        r.save_dir = str(save_dir)
        return r


@pytest.fixture
def small_dataset(tmp_path):
    root = tmp_path / "ds"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    for i in range(6):
        (root / "images" / "train" / f"im{i}.jpg").write_bytes(b"x")
        (root / "labels" / "train" / f"im{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "images" / "val" / "v.jpg").write_bytes(b"x")
    (root / "labels" / "val" / "v.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    with open(root / "data.yaml", 'w') as f:
        yaml.safe_dump({'path': str(root), 'train': 'images/train', 'val': 'images/val', 'names': ['a', 'b']}, f)
    return root


class TestOutOfFold:
    def test_no_image_is_predicted_by_a_model_that_trained_on_it(self, small_dataset, tmp_path):
        FakeTrainer.calls = []
        config = {'output_dir': str(tmp_path / 'out'), 'seed': 3, 'oof_folds': 3, 'oof_epochs': 1}
        with patch('src.detector.oof_models.YOLO', FakeTrainer):
            mapping = train_oof_models(str(small_dataset), config)

        assert len(mapping) == 6 and len(FakeTrainer.calls) == 3
        out = tmp_path / 'out' / 'oof'
        for image, weights in mapping.items():
            fold = Path(weights).parent.parent.name            # 'fold0', 'fold1', ...
            trained_on = (out / f"{fold}_train.txt").read_text().split()
            assert str(small_dataset / "images" / "train" / image) not in trained_on
        for i in range(3):                                      # and every other image WAS in each fold's training set
            held_out = {n for n, w in mapping.items() if f"fold{i}" in w}
            trained = {Path(p).name for p in (out / f"fold{i}_train.txt").read_text().split()}
            assert held_out.isdisjoint(trained) and held_out | trained == set(mapping)

    def test_training_arguments(self, small_dataset, tmp_path):
        FakeTrainer.calls = []
        config = {'output_dir': str(tmp_path / 'out'), 'oof_epochs': 4, 'device': 'cpu'}
        with patch('src.detector.oof_models.YOLO', FakeTrainer):
            train_oof_models(str(small_dataset), config)
        kw = FakeTrainer.calls[0]
        assert kw['epochs'] == 4 and kw['device'] == 'cpu' and kw['val'] is False
        assert Path(kw['project']).is_absolute()
        cfg = yaml.safe_load(Path(kw['data']).read_text())
        assert cfg['names'] == ['a', 'b'] and Path(cfg['val']).name == 'val'

    def test_old_folds_are_reused_only_when_resuming(self, small_dataset, tmp_path):
        config = {'output_dir': str(tmp_path / 'out'), 'oof_epochs': 1}
        with patch('src.detector.oof_models.YOLO', FakeTrainer):
            FakeTrainer.calls = []
            train_oof_models(str(small_dataset), config)
            assert len(FakeTrainer.calls) == 2
            FakeTrainer.calls = []
            train_oof_models(str(small_dataset), config)                      # data may have changed: retrain
            assert len(FakeTrainer.calls) == 2
            FakeTrainer.calls = []
            train_oof_models(str(small_dataset), {**config, 'resume': True})  # explicit resume: reuse
            assert FakeTrainer.calls == []


# ----------------------------------------------------------------------------- scoring
def _box(cls, x, y, w, h):
    return {'cls': cls, 'x': x, 'y': y, 'w': w, 'h': h}


@pytest.fixture
def scoring_inputs(tmp_path):
    A, R = _box(0, 0.2, 0.2, 0.2, 0.2), _box(1, 0.8, 0.2, 0.2, 0.2)
    O, S = _box(0, 0.5, 0.8, 0.2, 0.2), _box(0, 0.6, 0.9, 0.2, 0.2)
    D = _box(2, 0.3, 0.3, 0.1, 0.1)
    manifest = [
        {'file': 'labels/train/a.txt', 'line': 0, 'type': 'deleted', 'original': A, 'result': None},
        {'file': 'labels/train/b.txt', 'line': 0, 'type': 'class_flip', 'original': {**R, 'cls': 0}, 'result': R},
        {'file': 'labels/train/c.txt', 'line': 0, 'type': 'coordinate_shift', 'original': O, 'result': S,
         'iou_with_original': 0.3},
        {'file': 'labels/train/c.txt', 'line': 1, 'type': 'coordinate_shift', 'original': O, 'result': S,
         'iou_with_original': 0.9},                                                       # too mild to count
        {'file': 'labels/train/d.txt', 'line': 0, 'type': 'duplicate', 'original': D, 'result': D},
    ]
    audit = tmp_path / 'audit'
    audit.mkdir()
    (audit / 'missing_annotations.csv').write_text(
        "image,pred_class,confidence,pred_cx,pred_cy,pred_w,pred_h\n"
        "a.jpg,0,0.9,0.2,0.2,0.2,0.2\n"          # finds the deleted box, right class
        "z.jpg,1,0.9,0.5,0.5,0.1,0.1\n")         # not an injected error
    (audit / 'class_disagreements.csv').write_text(
        "image,gt_line,gt_class,pred_class,confidence,gt_cx,gt_cy,gt_w,gt_h\n"
        "b.jpg,1,1,0,0.9,0.8,0.2,0.2,0.2\n")
    (audit / 'localization_disagreements.csv').write_text(
        "image,gt_line,class,confidence,iou,gt_cx,gt_cy,gt_w,gt_h,pred_cx,pred_cy,pred_w,pred_h\n"
        "c.jpg,1,0,0.9,0.3,0.6,0.9,0.2,0.2,0.5,0.8,0.2,0.2\n")
    analysis = tmp_path / 'analysis.json'
    analysis.write_text(json.dumps({'suspicious_details': [
        {'image': 'd.jpg', 'issue': 'exact_duplicate', 'box': [0.3, 0.3, 0.1, 0.1], 'line': 2}]}))
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, audit, analysis


class TestScoring:
    def test_recall_and_precision(self, scoring_inputs, tmp_path):
        manifest, audit, analysis = scoring_inputs
        result = score_detector(str(manifest), str(audit), str(analysis), output_dir=str(tmp_path / 'eval'))
        r = result['recall']
        assert (r['deleted']['injected'], r['deleted']['caught'], r['deleted']['restored_with_correct_class']) == (1, 1, 1)
        assert (r['class_flip']['injected'], r['class_flip']['caught'], r['class_flip']['diagnosed_with_correct_class']) == (1, 1, 1)
        s = r['shift_or_resize']
        assert (s['injected'], s['caught'], s['predicted_box_matches_original'], s['excluded_mild']) == (1, 1, 1, 1)
        assert (r['duplicate']['injected'], r['duplicate']['caught']) == (1, 1)

        p = result['precision']
        assert (p['missing']['flags'], p['missing']['matched_injected_error']) == (2, 1)
        assert p['missing']['precision'] == 0.5
        assert p['class_disagreement']['precision'] == 1.0 and p['localization']['precision'] == 1.0

        assert (tmp_path / 'eval' / 'detector_eval.md').exists()
        assert json.loads((tmp_path / 'eval' / 'detector_eval.json').read_text())['recall']['deleted']['recall'] == 1.0

    def test_a_detector_that_flags_nothing_scores_zero_recall(self, scoring_inputs, tmp_path):
        manifest, _, _ = scoring_inputs
        empty = tmp_path / 'empty_audit'
        empty.mkdir()
        result = score_detector(str(manifest), str(empty))
        assert result['recall']['deleted']['recall'] == 0.0
        assert result['recall']['class_flip']['recall'] == 0.0
        assert result['recall']['duplicate']['recall'] is None          # no analysis report given
        assert result['precision']['missing']['precision'] is None      # no flags, so precision undefined

    def test_wrong_box_does_not_count_as_caught(self, scoring_inputs, tmp_path):
        manifest, audit, _ = scoring_inputs
        (audit / 'missing_annotations.csv').write_text(
            "image,pred_class,confidence,pred_cx,pred_cy,pred_w,pred_h\n"
            "a.jpg,0,0.9,0.9,0.9,0.1,0.1\n")                            # right image, wrong place
        assert score_detector(str(manifest), str(audit))['recall']['deleted']['caught'] == 0
