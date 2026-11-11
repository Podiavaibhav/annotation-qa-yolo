"""
Tests for the cleaner's three modes and its per-box fixes.
"""
import json
import yaml
import pytest
from pathlib import Path

from src.cleaner.dataset_cleaner import DatasetCleaner


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "ds"
    for d in ("images/train", "labels/train", "images/val", "labels/val"):
        (root / d).mkdir(parents=True)
    (root / "images/train/img1.jpg").write_bytes(b"x")
    (root / "images/train/img2.jpg").write_bytes(b"x")       # img2 has no label file at all
    (root / "images/val/v.jpg").write_bytes(b"x")
    (root / "labels/val/v.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "labels/train/img1.txt").write_text(
        "0 0.250000 0.250000 0.200000 0.200000\n"    # line 1: class is wrong (should be 1)
        "0 0.750000 0.250000 0.200000 0.200000\n"    # line 2: badly placed
        "1 0.250000 0.750000 0.200000 0.200000\n"    # line 3: fine
        "1 0.250000 0.750000 0.200000 0.200000\n"    # line 4: exact duplicate of line 3
        "0 0.900000 0.900000 0.000500 0.000500\n")   # line 5: tiny
    with open(root / "data.yaml", 'w') as f:
        yaml.safe_dump({'path': 'datasets/elsewhere', 'train': 'images/train', 'val': 'images/val',
                        'names': ['a', 'b']}, f)
    return root


@pytest.fixture
def reports(tmp_path):
    analysis = tmp_path / 'analysis.json'
    analysis.write_text(json.dumps({
        'invalid_details': [],
        'suspicious_details': [
            {'image': 'img1.jpg', 'issue': 'exact_duplicate', 'box': [0.25, 0.75, 0.2, 0.2], 'line': 4},
            {'image': 'img1.jpg', 'issue': 'overlapping_different_class', 'box': [0.75, 0.25, 0.2, 0.2], 'line': 2},
            {'image': 'img1.jpg', 'issue': 'tiny_box', 'box': [0.9, 0.9, 0.0005, 0.0005], 'line': 5},
        ]}))
    audit = tmp_path / 'audit'
    audit.mkdir()
    (audit / 'class_disagreements.csv').write_text(
        "image,gt_line,gt_class,pred_class,confidence,iou\n"
        "img1.jpg,1,0,1,0.95,0.9\n")
    (audit / 'localization_disagreements.csv').write_text(
        "image,gt_line,class,confidence,iou,pred_cx,pred_cy,pred_w,pred_h\n"
        "img1.jpg,2,0,0.9,0.2,0.8,0.3,0.2,0.2\n")
    (audit / 'missing_annotations.csv').write_text(
        "image,pred_class,confidence,pred_cx,pred_cy,pred_w,pred_h\n"
        "img2.jpg,1,0.9,0.5,0.5,0.3,0.3\n"        # img2 has no labels: add this
        "img2.jpg,0,0.5,0.1,0.1,0.1,0.1\n"        # not confident enough
        "img1.jpg,0,0.95,0.26,0.26,0.2,0.2\n")    # something already labels that object (line 1)
    return analysis, audit


def _clean(dataset, reports, tmp_path, mode):
    analysis, audit = reports
    config = {'output_dir': str(tmp_path / f'out_{mode}'), 'cleaning_mode': mode}
    cleaner = DatasetCleaner(str(dataset), config, analysis_report_path=str(analysis), model_audit_path=str(audit))
    counts = cleaner.clean()
    out = cleaner.cleaned_dataset_dir / 'labels' / 'train'
    read = lambda name: [l.split() for l in (out / name).read_text().splitlines()] if (out / name).exists() else None
    return counts, read('img1.txt'), read('img2.txt'), cleaner


class TestModes:
    def test_conservative_removes_only_the_duplicate_copy(self, dataset, reports, tmp_path):
        counts, img1, img2, _ = _clean(dataset, reports, tmp_path, 'conservative')
        assert counts == {'removed': 1, 'kept': 4, 'relabeled': 0, 'replaced': 0, 'added': 0}
        assert len(img1) == 4 and img1[2][1:3] == ['0.250000', '0.750000']     # first copy of the duplicate survives
        assert img1[0][0] == '0' and img2 == []                                 # nothing relabeled or added

    def test_model_assisted_fixes_boxes_one_by_one(self, dataset, reports, tmp_path):
        counts, img1, img2, _ = _clean(dataset, reports, tmp_path, 'model_assisted')
        assert counts == {'removed': 1, 'kept': 4, 'relabeled': 1, 'replaced': 1, 'added': 1}
        assert img1[0][0] == '1'                                                # class corrected
        assert [float(v) for v in img1[1][1:]] == [0.8, 0.3, 0.2, 0.2]          # box moved to the prediction
        assert img1[2][0] == '1' and len(img1) == 4                             # untouched box and one duplicate left
        assert img1[3] == ['0', '0.900000', '0.900000', '0.000500', '0.000500']      # tiny box is kept in this mode
        assert len(img2) == 1 and img2[0][0] == '1' and img2[0][1] == '0.500000'     # one confident box added

    def test_aggressive_also_removes_suspicious_boxes(self, dataset, reports, tmp_path):
        counts, img1, _, _ = _clean(dataset, reports, tmp_path, 'aggressive')
        assert counts['removed'] == 2 and counts['kept'] == 3                   # duplicate + tiny box
        assert all(r[3] != '0.000500' for r in img1)

    def test_one_disagreement_never_removes_other_boxes_of_the_same_class(self, dataset, reports, tmp_path):
        """Regression: the old cleaner keyed the audit on (image, class) and deleted every box of that class."""
        _, img1, _, _ = _clean(dataset, reports, tmp_path, 'model_assisted')
        assert len(img1) == 4

    def test_overlapping_different_class_is_not_auto_removed(self, dataset, reports, tmp_path):
        counts, _, _, cleaner = _clean(dataset, reports, tmp_path, 'aggressive')
        assert ('img1.jpg', 2) not in cleaner.suspicious_to_remove | cleaner.duplicates_to_remove


class TestOutputDataset:
    def test_original_is_untouched_and_yaml_is_absolute(self, dataset, reports, tmp_path):
        before = (dataset / "labels/train/img1.txt").read_text()
        _, _, _, cleaner = _clean(dataset, reports, tmp_path, 'model_assisted')
        assert (dataset / "labels/train/img1.txt").read_text() == before
        cfg = yaml.safe_load((cleaner.cleaned_dataset_dir / 'data.yaml').read_text())
        assert Path(cfg['path']) == cleaner.cleaned_dataset_dir.resolve()
        assert cfg['train'] == 'images/train' and cfg['val'] == 'images/val'
        assert (cleaner.cleaned_dataset_dir / 'labels/val/v.txt').read_text() == "0 0.5 0.5 0.2 0.2\n"

    def test_summary_and_manifest_written(self, dataset, reports, tmp_path):
        _, _, _, cleaner = _clean(dataset, reports, tmp_path, 'model_assisted')
        manifest = json.loads((cleaner.cleaning_report_dir / 'cleaning_manifest.json').read_text())
        assert {m['action'] for m in manifest} == {'removed', 'relabeled', 'box_replaced', 'box_added'}
        assert (cleaner.cleaning_report_dir / 'cleaning_summary.csv').exists()

    def test_rerun_starts_from_a_clean_folder(self, dataset, reports, tmp_path):
        _, _, _, cleaner = _clean(dataset, reports, tmp_path, 'conservative')
        stale = cleaner.cleaned_dataset_dir / 'labels/train/leftover.txt'
        stale.write_text("0 0.5 0.5 0.1 0.1\n")
        cleaner.clean()
        assert not stale.exists()

    def test_refuses_to_delete_the_source_dataset(self, tmp_path):
        out = tmp_path / "out"
        src = out / "cleaned_dataset"                  # the cleaner's output folder IS the dataset
        for d in ("images/train", "labels/train"):
            (src / d).mkdir(parents=True)
        (src / "images/train/a.jpg").write_bytes(b"x")
        (src / "labels/train/a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        with open(src / "data.yaml", 'w') as f:
            yaml.safe_dump({'path': '.', 'train': 'images/train', 'names': ['a']}, f)
        cleaner = DatasetCleaner(str(src), {'output_dir': str(out)})
        with pytest.raises(ValueError):
            cleaner.clean()
        assert (src / "labels/train/a.txt").exists()

    def test_unknown_mode_is_rejected(self, dataset, tmp_path):
        with pytest.raises(ValueError):
            DatasetCleaner(str(dataset), {'output_dir': str(tmp_path / 'o'), 'cleaning_mode': 'yolo'})

    def test_audit_rows_from_an_old_detector_are_ignored_not_crashing(self, dataset, tmp_path):
        audit = tmp_path / 'old_audit'
        audit.mkdir()
        (audit / 'class_disagreements.csv').write_text("image,gt_class,pred_class,confidence,iou\nimg1.jpg,0,1,0.9,0.8\n")
        (audit / 'missing_annotations.csv').write_text("image,pred_class,confidence\nimg2.jpg,1,0.9\n")
        cleaner = DatasetCleaner(str(dataset), {'output_dir': str(tmp_path / 'o'), 'cleaning_mode': 'model_assisted'},
                                 model_audit_path=str(audit))
        assert cleaner.relabel == {} and not cleaner.add
