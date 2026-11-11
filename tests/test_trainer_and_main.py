"""
Tests for trainer/evaluator arguments, multi-seed aggregation, the report, and `prepare`.
"""
import sys
import types
import yaml
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
from src.training.evaluator import evaluate_model
from src.training.report_generator import ReportGenerator
from src.training.trainer import device_kwargs, train_model


class Recorder:
    last = {}

    def __init__(self, *a, **k):
        pass

    def train(self, **kw):
        Recorder.last = kw
        save = Path(kw['project']) / kw['name']
        (save / 'weights').mkdir(parents=True, exist_ok=True)
        (save / 'weights' / 'best.pt').write_text('x')
        return SimpleNamespace(save_dir=str(save))

    def val(self, **kw):
        Recorder.last = kw
        return SimpleNamespace(box=SimpleNamespace(map50=0.5, map=0.3, mp=0.6, mr=0.4))


class TestTrainerArguments:
    def test_seed_override_and_reproducibility_flags(self, tmp_path):
        with patch('src.training.trainer.YOLO', Recorder):
            train_model('d.yaml', str(tmp_path / 'r'), {'seed': 42}, seed=7)
        assert Recorder.last['seed'] == 7 and Recorder.last['deterministic'] is True

    def test_config_seed_is_the_default(self, tmp_path):
        with patch('src.training.trainer.YOLO', Recorder):
            train_model('d.yaml', str(tmp_path / 'r'), {'seed': 42})
        assert Recorder.last['seed'] == 42

    def test_project_is_absolute_so_nothing_lands_in_runs_detect(self, tmp_path):
        import os
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch('src.training.trainer.YOLO', Recorder):
                path = train_model('d.yaml', 'relative/dir', {})
        finally:
            os.chdir(old)
        assert Path(Recorder.last['project']).is_absolute() and path.is_absolute()

    def test_device_only_passed_when_configured(self, tmp_path):
        assert device_kwargs({}) == {} and device_kwargs({'device': 'auto'}) == {}
        assert device_kwargs({'device': 0}) == {'device': 0}
        with patch('src.training.trainer.YOLO', Recorder):
            train_model('d.yaml', str(tmp_path / 'a'), {})
            assert 'device' not in Recorder.last
            train_model('d.yaml', str(tmp_path / 'b'), {'device': 'cpu'})
            assert Recorder.last['device'] == 'cpu'

    def test_evaluator_uses_absolute_project_and_given_size(self, tmp_path):
        with patch('src.training.evaluator.YOLO', Recorder):
            out = evaluate_model('m.pt', 'd.yaml', str(tmp_path / 'e'), imgsz=320, batch=4)
        assert out == {'mAP50': 0.5, 'mAP50-95': 0.3, 'precision': 0.6, 'recall': 0.4}
        assert Recorder.last['imgsz'] == 320 and Path(Recorder.last['project']).is_absolute()


class TestAggregate:
    def test_mean_and_sample_std(self):
        runs = [{'mAP50': 0.4, 'mAP50-95': 0.2, 'precision': 0.5, 'recall': 0.3},
                {'mAP50': 0.6, 'mAP50-95': 0.4, 'precision': 0.7, 'recall': 0.5}]
        agg = main.aggregate(runs)
        assert agg['n_seeds'] == 2 and agg['mAP50'] == pytest.approx(0.5)
        assert agg['mAP50_std'] == pytest.approx(0.14142, abs=1e-4)      # sample std (n-1)

    def test_single_seed_has_zero_std(self):
        agg = main.aggregate([{'mAP50': 0.4, 'mAP50-95': 0.2, 'precision': 0.5, 'recall': 0.3}])
        assert agg['mAP50_std'] == 0.0 and agg['n_seeds'] == 1


def _results(original, noisy, cleaned, std):
    mk = lambda m: {'mAP50': m, 'mAP50-95': m * .6, 'precision': m, 'recall': m, 'n_seeds': 3,
                    'mAP50_std': std, 'mAP50-95_std': std, 'precision_std': std, 'recall_std': std}
    return {'Original': mk(original), 'Noisy': mk(noisy), 'Cleaned': mk(cleaned)}


class TestReport:
    def test_shows_std_and_share_recovered(self, tmp_path):
        gen = ReportGenerator(str(tmp_path))
        results = _results(0.80, 0.60, 0.74, 0.01)
        gen.generate_comparison(results)
        gen.generate_markdown_report(results)
        text = (gen.output_dir / 'final_report.md').read_text()
        assert '0.8000 ± 0.0100' in text and 'over 3 training seeds' in text
        assert 'Share of the loss recovered: 70.0%' in text
        assert 'within one standard deviation' not in text
        assert (gen.output_dir / 'performance_comparison.png').exists()

    def test_warns_when_the_gain_is_inside_seed_noise(self, tmp_path):
        gen = ReportGenerator(str(tmp_path))
        gen.generate_markdown_report(_results(0.80, 0.60, 0.61, 0.02))
        assert 'within one standard deviation' in (gen.output_dir / 'final_report.md').read_text()

    def test_old_single_run_results_still_work(self, tmp_path):
        gen = ReportGenerator(str(tmp_path))
        plain = {k: {'mAP50': v, 'mAP50-95': v / 2, 'precision': v, 'recall': v}
                 for k, v in (('Original', .8), ('Noisy', .6), ('Cleaned', .7))}
        gen.generate_comparison(plain)
        gen.generate_markdown_report(plain)
        assert '±' not in (gen.output_dir / 'final_report.md').read_text()


class TestPrepare:
    def test_prepare_writes_a_usable_yaml_and_remembers_it(self, tmp_path):
        root = tmp_path / "african-wildlife"
        (root / "train" / "images").mkdir(parents=True)
        info = {'path': root, 'train': root / 'train' / 'images', 'val': root / 'valid' / 'images',
                'names': {0: 'buffalo', 1: 'zebra'}}
        fake_utils = types.ModuleType('ultralytics.data.utils')
        fake_utils.check_det_dataset = lambda name: info
        modules = {'ultralytics.data': types.ModuleType('ultralytics.data'), 'ultralytics.data.utils': fake_utils}
        with patch.dict(sys.modules, modules):
            main.cmd_prepare(SimpleNamespace(dataset='african-wildlife.yaml'), {'output_dir': str(tmp_path / 'out')})
        written = Path((tmp_path / 'out' / 'prepared_dataset.txt').read_text())
        cfg = yaml.safe_load(written.read_text())
        assert written == root.resolve() / 'data.yaml' and cfg['nc'] == 2
