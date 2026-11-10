"""
Main CLI entry point for the Annotation QA & YOLO Retraining Pipeline.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from src.analyzer.annotation_analyzer import AnnotationAnalyzer
from src.cleaner.dataset_cleaner import DatasetCleaner
from src.detector.model_disagreement import ModelDisagreementDetector
from src.training.evaluator import evaluate_model
from src.training.noise_injector import NoiseInjector
from src.training.report_generator import ReportGenerator
from src.training.trainer import train_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found at {config_path}. Using defaults.")
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def find_yaml_file(directory: Path) -> Path:
    yamls = list(directory.glob('*.yaml'))
    if not yamls:
        raise FileNotFoundError(f"No .yaml file found in {directory}")
    return yamls[0]

def cmd_analyze(args, config):
    logger.info("Running Module 1: Dataset Quality Analyzer")
    analyzer = AnnotationAnalyzer(args.data, config)
    report = analyzer.analyze()
    report_path = Path(config.get('output_dir', 'outputs')) / 'analysis' / 'annotation_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Analysis report saved to {report_path}")

def cmd_detect(args, config):
    logger.info("Running Module 2: Model-Assisted Annotation Error Detection")
    model_name = getattr(args, 'model', None) or config.get('model_name', 'yolov8n.pt')
    detector = ModelDisagreementDetector(args.data, config, model_name=model_name)
    detector.detect()

def cmd_clean(args, config):
    logger.info("Running Module 3: Dataset Cleaning Engine")
    output_dir = Path(config.get('output_dir', 'outputs'))
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
    injector = NoiseInjector(args.data, config)
    noisy_path = injector.inject()
    logger.info(f"Noisy dataset created at {noisy_path}")

def cmd_train(args, config):
    """Trains and evaluates 3 models: Original, Noisy, and Cleaned."""
    logger.info("Running Module 4: Training, Evaluation, and Reporting")
    output_dir = Path(config.get('output_dir', 'outputs'))
    
    # Paths to the three datasets
    original_yaml = args.data
    noisy_dir = output_dir / 'noisy_dataset'
    noisy_yaml = find_yaml_file(noisy_dir)
    cleaned_dir = output_dir / 'cleaned_dataset'
    cleaned_yaml = find_yaml_file(cleaned_dir)
    
    train_dir = output_dir / 'experiments' / 'training_runs'
    models = {}
    
    logger.info("Training Original model...")
    models['Original'] = train_model(str(original_yaml), str(train_dir / 'original'), config)
    
    logger.info("Training Noisy model...")
    models['Noisy'] = train_model(str(noisy_yaml), str(train_dir / 'noisy'), config)
    
    logger.info("Training Cleaned model...")
    models['Cleaned'] = train_model(str(cleaned_yaml), str(train_dir / 'cleaned'), config)
    
    eval_dir = output_dir / 'experiments' / 'evaluations'
    results = {}
    logger.info("Evaluating all models on the ORIGINAL validation set...")
    for name, model_path in models.items():
        results[name] = evaluate_model(str(model_path), str(original_yaml), str(eval_dir / name))
        
    reporter = ReportGenerator(str(output_dir))
    reporter.generate_comparison(results)
    reporter.generate_markdown_report(results)
    logger.info("Module 4 complete.")

def cmd_run_all(args, config):
    logger.info("Starting Full Pipeline (run-all)")
    output_dir = Path(config.get('output_dir', 'outputs'))
    
    # 1. Analyze Original
    cmd_analyze(args, config)
    
    # 2. Detect on Original
    cmd_detect(args, config)
    
    # 3. Inject Noise (creates outputs/noisy_dataset)
    injector = NoiseInjector(args.data, config)
    noisy_dir = injector.inject()
    
    # 4. Analyze Noisy Dataset
    logger.info("Analyzing noisy dataset...")
    noisy_yaml = find_yaml_file(noisy_dir)
    analyzer = AnnotationAnalyzer(str(noisy_yaml), config)
    noisy_report = analyzer.analyze()
    noisy_report_path = output_dir / 'analysis' / 'noisy_annotation_report.json'
    with open(noisy_report_path, 'w') as f:
        json.dump(noisy_report, f, indent=2)
        
    # 5. Clean Noisy Dataset (creates outputs/cleaned_dataset)
    logger.info("Cleaning noisy dataset...")
    cleaner = DatasetCleaner(str(noisy_yaml), config, analysis_report_path=str(noisy_report_path))
    cleaner.clean()
    
    # 6. Train and Evaluate
    # We temporarily override args.data to point to original for the train command's evaluation baseline
    original_data = args.data
    cmd_train(args, config)
    
    logger.info("Full Pipeline complete.")

def main():
    parser = argparse.ArgumentParser(description="Annotation QA & YOLO Retraining Pipeline")
    parser.add_argument('--config', default='configs/default.yaml', help='Path to configuration file')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    p_analyze = subparsers.add_parser('analyze', help='Run Module 1')
    p_analyze.add_argument('--data', required=True, help='Path to dataset directory or YAML file')
    
    p_detect = subparsers.add_parser('detect', help='Run Module 2')
    p_detect.add_argument('--data', required=True, help='Path to dataset directory or YAML file')
    p_detect.add_argument('--model', default=None, help='Path to YOLO model weights')
    
    p_clean = subparsers.add_parser('clean', help='Run Module 3')
    p_clean.add_argument('--data', required=True, help='Path to dataset directory or YAML file')
    
    p_noise = subparsers.add_parser('inject-noise', help='Create a synthetic noisy dataset')
    p_noise.add_argument('--data', required=True, help='Path to dataset directory or YAML file')
    
    p_train = subparsers.add_parser('train', help='Run Module 4: Training and Evaluation')
    p_train.add_argument('--data', required=True, help='Path to ORIGINAL dataset YAML (for eval baseline)')
    
    p_all = subparsers.add_parser('run-all', help='Execute the complete pipeline')
    p_all.add_argument('--data', required=True, help='Path to dataset directory or YAML file')
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    config = load_config(args.config)
    commands = {
        'analyze': cmd_analyze, 'detect': cmd_detect, 'clean': cmd_clean,
        'inject-noise': cmd_inject_noise, 'train': cmd_train, 'run-all': cmd_run_all
    }
    commands[args.command](args, config)

if __name__ == '__main__':
    main()
