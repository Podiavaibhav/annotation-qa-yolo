"""
End-to-end test of `main.cmd_run_all` with a fake YOLO instead of Ultralytics.

What the fake does, so the test can say something real about the pipeline:
  * It resolves dataset yaml files the way Ultralytics does: a relative `path:` is looked up in
    Ultralytics' own datasets folder. So the Kaggle bug ("images not found") reproduces here
    unless every yaml the pipeline hands over has an absolute path.
  * Inference is an "oracle with blind spots": it predicts the TRUE boxes confidently but misses
    about one box in seven, standing in for a decent out-of-fold model.
  * "Training" scores the labels it was given against the truth, and evaluation returns that
    score, so Original > Cleaned > Noisy only if cleaning really repaired the labels.

This checks plumbing and logic. It cannot say anything about real mAP; that needs a real run.
"""
import json
import yaml
import cv2
import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main

TRUTH = {}          # image name -> [(cls, cx, cy, w, h)] of the ORIGINAL labels
TRAIN_YAMLS = []    # every data yaml handed to training


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def _label_quality(data_yaml):
    cfg = yaml.safe_load(open(data_yaml))
    path = Path(cfg.get('path', ''))
    if not path.is_absolute():
        path = Path('/ultralytics_datasets_dir_that_has_nothing') / path       # what Ultralytics does
    if str(cfg['train']).endswith('.txt'):
        return None                                                              # an out-of-fold helper model
    images = path / cfg['train']
    if not images.exists():
        raise FileNotFoundError(f"Dataset '{data_yaml}' images not found, missing path '{images}'")
    labels = Path(str(images).replace('/images/', '/labels/'))
    matched = total = 0
    for name, truth in TRUTH.items():
        f = labels / (Path(name).stem + '.txt')
        cur = [(int(l.split()[0]), *map(float, l.split()[1:])) for l in f.read_text().splitlines()] if f.exists() else []
        used = set()
        for t in truth:
            for j, c in enumerate(cur):
                if j not in used and c[0] == t[0] and _iou(t[1:], c[1:]) >= 0.7:
                    used.add(j)
                    matched += 1
                    break
        total += max(len(truth), len(cur))
    return matched / total


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


class FakeYOLO:
    def __init__(self, weights='yolov8n.pt'):
        self.quality = None
        p = Path(str(weights))
        if p.exists() and (p.parent / 'quality.json').exists():
            self.quality = json.loads((p.parent / 'quality.json').read_text())

    def __call__(self, img_path, **kw):
        name = Path(img_path).name
        preds = []
        for k, (cls, cx, cy, w, h) in enumerate(TRUTH.get(name, [])):
            if (int(name[2:-4]) + k) % 7 == 0:
                continue                                                          # the model's blind spot
            preds.append(([(cx - w / 2) * 100, (cy - h / 2) * 100, (cx + w / 2) * 100, (cy + h / 2) * 100], 0.9, cls))
        return [SimpleNamespace(boxes=_Boxes(preds))]

    def train(self, data, project, name, **kw):
        TRAIN_YAMLS.append(str(data))
        save_dir = Path(project) / name
        w = save_dir / 'weights'
        w.mkdir(parents=True, exist_ok=True)
        (w / 'best.pt').write_text('b')
        (w / 'last.pt').write_text('l')
        (w / 'quality.json').write_text(json.dumps(_label_quality(data)))
        return SimpleNamespace(save_dir=str(save_dir))

    def val(self, **kw):
        q = self.quality
        return SimpleNamespace(box=SimpleNamespace(map50=q, map=0.6 * q, mp=q, mr=q))


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "synthetic"
    for d in ("images/train", "labels/train", "images/val", "labels/val"):
        (root / d).mkdir(parents=True)
    rng = np.random.default_rng(11)
    TRUTH.clear()
    for split, n in (("train", 60), ("val", 8)):
        for i in range(n):
            name = f"im{i}.jpg" if split == "train" else f"va{i}.jpg"
            cv2.imwrite(str(root / "images" / split / name), np.zeros((100, 100, 3), np.uint8))
            lines, boxes = [], []
            for cx, cy in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]:
                if rng.random() < 0.8:
                    box = (int(rng.integers(0, 3)), cx + rng.uniform(-.03, .03), cy + rng.uniform(-.03, .03), 0.22, 0.22)
                    boxes.append(box)
                    lines.append("%d %.6f %.6f %.6f %.6f" % box)
            (root / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n")
            if split == "train":
                TRUTH[name] = boxes
    with open(root / "data.yaml", 'w') as f:
        # the same mistake as coco8.yaml: a relative path that points nowhere
        yaml.safe_dump({'path': 'datasets/synthetic', 'train': 'images/train', 'val': 'images/val',
                        'names': ['a', 'b', 'c']}, f)
    return root


def _run(dataset, tmp_path, **overrides):
    config = {'output_dir': str(tmp_path / 'out'), 'seed': 5, 'train_seeds': [1, 2], 'epochs': 1,
              'oof_folds': 2, 'oof_epochs': 1, 'detector_mode': 'oof', 'cleaning_mode': 'model_assisted',
              'confidence_threshold': 0.5, **overrides}
    args = SimpleNamespace(data=str(dataset / "data.yaml"), resume=False)
    TRAIN_YAMLS.clear()
    patches = [patch(f'src.{m}.YOLO', FakeYOLO) for m in
               ('detector.model_disagreement', 'detector.oof_models', 'training.trainer', 'training.evaluator')]
    for p in patches:
        p.start()
    try:
        main.cmd_run_all(args, config)
    finally:
        for p in patches:
            p.stop()
    return Path(config['output_dir'])


class TestRunAll:
    def test_full_pipeline_runs_and_cleaning_repairs_the_labels(self, dataset, tmp_path):
        out = _run(dataset, tmp_path)
        results = json.loads((out / 'experiments' / 'comparison.json').read_text())

        original, noisy, cleaned = (results[k]['mAP50'] for k in ('Original', 'Noisy', 'Cleaned'))
        print(f"\n  label quality  original={original:.3f}  noisy={noisy:.3f}  cleaned={cleaned:.3f}")
        assert original == pytest.approx(1.0)
        assert noisy < 0.85                          # the noise really hurt
        assert cleaned > noisy + 0.08                # and cleaning really repaired a good part of it
        assert cleaned <= original + 1e-9
        assert results['Noisy']['n_seeds'] == 2

        per_seed = json.loads((out / 'experiments' / 'results_per_seed.json').read_text())
        assert all(len(v) == 2 for v in per_seed.values())
        report = (out / 'experiments' / 'final_report.md').read_text()
        assert '±' in report and 'recovered' in report

    def test_every_yaml_given_to_training_has_an_absolute_path(self, dataset, tmp_path):
        """Regression for the Kaggle crash: a relative `path:` makes Ultralytics look in the wrong folder."""
        out = _run(dataset, tmp_path)
        yamls = [Path(y) for y in TRAIN_YAMLS if not y.endswith(('fold0.yaml', 'fold1.yaml'))]
        assert len(yamls) == 6                                   # 3 variants x 2 seeds
        for y in yamls:
            assert Path(yaml.safe_load(y.read_text())['path']).is_absolute()
        trained_on = {Path(yaml.safe_load(y.read_text())['path']).name for y in yamls}
        assert trained_on == {'synthetic', 'noisy_dataset', 'cleaned_dataset'}

    def test_detector_is_scored_against_the_injected_noise(self, dataset, tmp_path):
        out = _run(dataset, tmp_path)
        ev = json.loads((out / 'experiments' / 'detector_eval.json').read_text())
        r = ev['recall']
        print("\n  recall:", {k: (v['caught'], v['injected']) for k, v in r.items()})
        assert r['class_flip']['recall'] > 0.6 and r['shift_or_resize']['recall'] > 0.6
        assert r['deleted']['recall'] > 0.4 and r['duplicate']['recall'] == 1.0
        assert ev['precision']['class_disagreement']['precision'] > 0.9
        assert (out / 'experiments' / 'detector_eval.md').exists()

    def test_detection_ran_on_the_noisy_data_with_out_of_fold_models(self, dataset, tmp_path):
        out = _run(dataset, tmp_path)
        mapping = json.loads((out / 'oof' / 'oof_models.json').read_text())
        assert len(mapping) == 60 and len({*mapping.values()}) == 2
        # fold training lists point at the NOISY copy, so the helper models learned from the noisy labels
        listed = (out / 'oof' / 'fold0_train.txt').read_text().split()[0]
        assert 'noisy_dataset' in listed

    def test_pretrained_mode_skips_fold_training(self, dataset, tmp_path):
        out = _run(dataset, tmp_path, detector_mode='pretrained')
        assert not (out / 'oof').exists()
        assert (out / 'model_audit').exists()

    def test_relative_path_bug_would_have_failed(self, dataset, tmp_path):
        """Prove the fake is strict: handing it the ORIGINAL yaml untouched fails like Kaggle did."""
        with pytest.raises(FileNotFoundError):
            _label_quality(dataset / "data.yaml")
