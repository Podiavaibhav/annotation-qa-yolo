"""
Final Reporting Pipeline.
Generates comparison tables, charts, and markdown reports.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates final experiment reports and comparisons."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir) / 'experiments'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_comparison(self, results: Dict[str, Dict]) -> None:
        """Generate CSV, JSON, and PNG comparison of experiment results."""
        df = pd.DataFrame.from_dict(results, orient='index')
        df.index.name = 'Experiment'
        
        csv_path = self.output_dir / 'comparison.csv'
        df.to_csv(csv_path)
        
        json_path = self.output_dir / 'comparison.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        plt.figure(figsize=(10, 6))
        df[['mAP50', 'mAP50-95']].plot(kind='bar', figsize=(10, 6))
        plt.title('Model Performance Comparison')
        plt.ylabel('Metric Value')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=150)
        plt.close()

        logger.info(f"Comparison reports saved to {self.output_dir}")

    def generate_markdown_report(self, results: Dict[str, Dict], analysis_stats: Dict = None) -> None:
        """Generate a final Markdown report."""
        report_path = self.output_dir / 'final_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# YOLO Annotation QA Experiment Report\n\n")
            f.write("## 1. Experiment Results\n\n")
            f.write("| Experiment | mAP50 | mAP50-95 | Precision | Recall |\n")
            f.write("|---|---|---|---|---|\n")
            for exp, metrics in results.items():
                f.write(f"| {exp} | {metrics['mAP50']:.4f} | {metrics['mAP50-95']:.4f} | "
                        f"{metrics['precision']:.4f} | {metrics['recall']:.4f} |\n")
            
            f.write("\n## 2. Analysis\n\n")
            if 'Original' in results and 'Noisy' in results and 'Cleaned' in results:
                loss = results['Original']['mAP50'] - results['Noisy']['mAP50']
                recovery = results['Cleaned']['mAP50'] - results['Noisy']['mAP50']
                f.write(f"- Accuracy lost due to noise: {loss:.4f}\n")
                f.write(f"- Accuracy recovered by cleaning: {recovery:.4f}\n")

        logger.info(f"Markdown report saved to {report_path}")
