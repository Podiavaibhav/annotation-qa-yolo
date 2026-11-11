"""
Main CLI entry point for the Annotation QA & YOLO Retraining Pipeline.
"""
import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import yaml

from src.analyzer.annotation_analyzer import AnnotationAnalyzer
from src.cleaner.dataset_cleaner import DatasetCleaner
from src.detector.model_disagreement import ModelDisagreementDetector
from src.detector.oof_models import train_oof_models
from src.detector.scoring import score_detector
from src.training.evaluator import evaluate_model
from src.training.noise_injector import NoiseInjector
from src.training.report_generator import ReportGenerator
from src.training.trainer import train_model
from src.yaml_utils import absolutize_yaml, write_prepared_yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

METRICS = ['mAP50', 'mAP50-95', 'precision', 'recall']


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found at {config_path}. Using defaults.")
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def find_yaml_file(directory: Path) -> Path:
    yamls = list(Path(directory).glob('*.yaml'))
    if not yamls:
        raise FileNotFoundError(f"No .yaml file found in {directory}")
    return yamls[0]


def out_dir_of(config: dict) -> Path:
    return Path(config.get('output_dir', 'outputs'))


def aggregate(runs: List[Dict]) -> Dict:
    """Mean and standard deviation of each metric over several training seeds."""
    result = {'n_seeds': len(runs)}
    for m in METRICS:
        values = [r[m] for r in runs]
        result[m] = statistics.fmean(values)
        result[f'{m}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result


def cmd_prepare(args, config):
    """Download a dataset with Ultralytics and write a yaml the rest of the pipeline can read."""
    from ultralytics.data.utils import check_det_dataset
    info = check_det_dataset(args.dataset)
    yaml_path = write_prepared_yaml(info)
    marker = out_dir_of(config) / 'prepared_dataset.txt'
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(yaml_path))
    logger.info(f"Dataset ready. Use:  --data {yaml_path}")


def cmd_analyze(args, config):
    logger.info("Running Module 1: Dataset Quality Analyzer")
    analyzer = AnnotationAnalyzer(args.data, config)
    report = analyzer.analyze()
    report_path = out_dir_of(config) / 'analysis' / 'annotation_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Analysis report saved to {report_path}")


def run_detection(data: str, config: dict, oof: bool, model_name: str = None):
    """Detect label errors. With oof=True every image is predicted by a model that never trained on it."""
    model_name = model_name or config.get('model_name', 'yolov8n.pt')
    fold_models = train_oof_models(data, config) if oof else None
    detector = ModelDisagreementDetector(data, config, model_name=model_name, fold_models=fold_models)
    return detector.detect()


def cmd_detect(args, config):
    logger.info("Running Module 2: Model-Assisted Annotation Error Detection")
    run_detection(args.data, config, oof=getattr(args, 'oof', False), model_name=getattr(args, 'model', None))


def cmd_clean(args, config):
    logger.info("Running Module 3: Dataset Cleaning Engine")
    output_dir = out_dir_of(config)
    analysis_report = output_dir / 'analysis' / 'annotation_report.json'
    model_audit = output_dir / 'model_audit'
    cleaner = DatasetCleaner(
        args.data, config,
        analysis_report_path=str(analysis_report) if analysis_report.exists() else None,
        model_audit_path=str(model_audit) if model_audit.exists() else None
    )
    cleaner.clean()


def cmd_inject_noise(args, config):
    logger.info("Running Synthetic Noise Injection")
    noisy_path = NoiseInjector(args.data, config).inject()
    logger.info(f"Noisy dataset created at {noisy_path}")


def cmd_score(args, config):
    """Score the detector against the noise that was injected."""
    output_dir = out_dir_of(config)
    manifest = output_dir / 'noisy_dataset' / 'corruption_manifest.json'
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found. Run inject-noise (or run-all) first.")
    analysis = output_dir / 'analysis' / 'noisy_annotation_report.json'
    score_detector(str(manifest), str(output_dir / 'model_audit'),
                   analysis_report_path=str(analysis) if analysis.exists() else None,
                   output_dir=str(output_dir / 'experiments'))
    logger.info(f"Detector evaluation saved to {output_dir / 'experiments' / 'detector_eval.md'}")


def cmd_train(args, config):
    """Trains and evaluates Original, Noisy and Cleaned models, each over several seeds."""
    logger.info("Running Module 4: Training, Evaluation, and Reporting")
    output_dir = out_dir_of(config)

    # Ultralytics needs an absolute dataset path. The original yaml often has a relative one.
    original_yaml = absolutize_yaml(args.data, output_dir / 'original_dataset.yaml')
    variants = {
        'Original': original_yaml,
        'Noisy': find_yaml_file(output_dir / 'noisy_dataset'),
        'Cleaned': find_yaml_file(output_dir / 'cleaned_dataset'),
    }
    seeds = config.get('train_seeds') or [config.get('seed', 42)]
    resume = getattr(args, 'resume', False)
    train_dir = output_dir / 'experiments' / 'training_runs'
    eval_dir = output_dir / 'experiments' / 'evaluations'

    per_seed = {name: [] for name in variants}
    for name, yaml_path in variants.items():
        for seed in seeds:
            run_dir = train_dir / name.lower() / f'seed{seed}'
            cache = run_dir / 'metrics.json'
            if resume and cache.exists():
                logger.info(f"{name} seed {seed}: reusing {cache}")
                per_seed[name].append(json.loads(cache.read_text()))
                continue
            logger.info(f"Training {name} model (seed {seed})...")
            model_path = train_model(str(yaml_path), str(run_dir), config, seed=seed)
            # Every model is judged on the ORIGINAL, untouched validation set.
            metrics = evaluate_model(str(model_path), str(original_yaml), str(eval_dir / name / f'seed{seed}'),
                                     imgsz=config.get('img_size', 640), batch=config.get('batch_size', 16))
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(metrics, indent=2))
            per_seed[name].append(metrics)

    results = {name: aggregate(runs) for name, runs in per_seed.items()}
    (output_dir / 'experiments').mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'experiments' / 'results_per_seed.json', 'w') as f:
        json.dump(per_seed, f, indent=2)

    reporter = ReportGenerator(str(output_dir))
    reporter.generate_comparison(results)
    reporter.generate_markdown_report(results)
    logger.info("Module 4 complete.")


def cmd_run_all(args, config):
    logger.info("Starting Full Pipeline (run-all)")
    output_dir = out_dir_of(config)
    config['resume'] = getattr(args, 'resume', False)

    # 1. Analyze the original dataset
    cmd_analyze(args, config)

    # 2. Inject noise (creates outputs/noisy_dataset and its manifest)
    noisy_dir = NoiseInjector(args.data, config).inject()
    noisy_yaml = find_yaml_file(noisy_dir)

    # 3. Analyze the noisy dataset
    logger.info("Analyzing noisy dataset...")
    noisy_report = AnnotationAnalyzer(str(noisy_yaml), config).analyze()
    noisy_report_path = output_dir / 'analysis' / 'noisy_annotation_report.json'
    with open(noisy_report_path, 'w') as f:
        json.dump(noisy_report, f, indent=2)

    # 4. Detect label errors in the NOISY data (the data we would actually have to clean)
    logger.info("Running Module 2 on the noisy dataset...")
    run_detection(str(noisy_yaml), config, oof=config.get('detector_mode', 'oof') == 'oof')

    # 5. Score the detector against what was really corrupted
    cmd_score(args, config)

    # 6. Clean the noisy dataset using both reports (creates outputs/cleaned_dataset)
    logger.info("Cleaning noisy dataset...")
    DatasetCleaner(str(noisy_yaml), config, analysis_report_path=str(noisy_report_path),
                   model_audit_path=str(output_dir / 'model_audit')).clean()

    # 7. Train and evaluate Original / Noisy / Cleaned
    cmd_train(args, config)
    logger.info("Full Pipeline complete.")


def main():
    parser = argparse.ArgumentParser(description="Annotation QA & YOLO Retraining Pipeline")
    parser.add_argument('--config', default='configs/default.yaml', help='Path to configuration file')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    p_prep = subparsers.add_parser('prepare', help='Download a dataset and write a usable yaml')
    p_prep.add_argument('--dataset', default='african-wildlife.yaml', help='Ultralytics dataset yaml name or path')

    p_analyze = subparsers.add_parser('analyze', help='Run Module 1')
    p_analyze.add_argument('--data', required=True, help='Path to dataset directory or YAML file')

    p_detect = subparsers.add_parser('detect', help='Run Module 2')
    p_detect.add_argument('--data', required=True, help='Path to dataset directory or YAML file')
    p_detect.add_argument('--model', default=None, help='Path to YOLO model weights')
    p_detect.add_argument('--oof', action='store_true',
                          help='Predict each image with a model that did not train on it (trains models first)')

    p_clean = subparsers.add_parser('clean', help='Run Module 3')
    p_clean.add_argument('--data', required=True, help='Path to dataset directory or YAML file')

    p_noise = subparsers.add_parser('inject-noise', help='Create a synthetic noisy dataset')
    p_noise.add_argument('--data', required=True, help='Path to dataset directory or YAML file')

    subparsers.add_parser('score', help='Score the detector against the injected noise')

    p_train = subparsers.add_parser('train', help='Run Module 4: Training and Evaluation')
    p_train.add_argument('--data', required=True, help='Path to ORIGINAL dataset YAML (for eval baseline)')
    p_train.add_argument('--resume', action='store_true', help='Reuse finished runs from an interrupted session')

    p_all = subparsers.add_parser('run-all', help='Execute the complete pipeline')
    p_all.add_argument('--data', required=True, help='Path to dataset directory or YAML file')
    p_all.add_argument('--resume', action='store_true', help='Reuse finished runs from an interrupted session')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)
    commands = {
        'prepare': cmd_prepare, 'analyze': cmd_analyze, 'detect': cmd_detect, 'clean': cmd_clean,
        'inject-noise': cmd_inject_noise, 'score': cmd_score, 'train': cmd_train, 'run-all': cmd_run_all
    }
    commands[args.command](args, config)


if __name__ == '__main__':
    main()
