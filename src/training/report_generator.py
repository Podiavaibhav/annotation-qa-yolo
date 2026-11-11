"""
Final Reporting Pipeline.
Generates comparison tables, charts, and markdown reports.

A result entry looks like {'mAP50': 0.61, 'mAP50-95': 0.40, 'precision': ..., 'recall': ...}.
When a variant was trained with several seeds, add '<metric>_std' keys and 'n_seeds' and the
reports show mean +/- std.
"""
import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

METRICS = ['mAP50', 'mAP50-95', 'precision', 'recall']


class ReportGenerator:
    """Generates final experiment reports and comparisons."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir) / 'experiments'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_comparison(self, results: Dict[str, Dict]) -> None:
        """CSV, JSON and a bar chart (with error bars when std is available)."""
        df = pd.DataFrame.from_dict(results, orient='index')
        df.index.name = 'Experiment'
        df.to_csv(self.output_dir / 'comparison.csv')
        with open(self.output_dir / 'comparison.json', 'w') as f:
            json.dump(results, f, indent=2)

        cols = ['mAP50', 'mAP50-95']
        fig, ax = plt.subplots(figsize=(10, 6))
        yerr = None
        if all(f'{c}_std' in df.columns for c in cols):
            yerr = df[[f'{c}_std' for c in cols]].to_numpy().T
        df[cols].plot(kind='bar', ax=ax, yerr=yerr, capsize=4)
        ax.set_title('Model Performance Comparison')
        ax.set_ylabel('Metric Value')
        plt.setp(ax.get_xticklabels(), rotation=0)
        fig.tight_layout()
        fig.savefig(self.output_dir / 'performance_comparison.png', dpi=150)
        plt.close(fig)
        logger.info(f"Comparison reports saved to {self.output_dir}")

    @staticmethod
    def _fmt(metrics: Dict, key: str) -> str:
        std = metrics.get(f'{key}_std')
        return f"{metrics[key]:.4f} ± {std:.4f}" if std is not None else f"{metrics[key]:.4f}"

    def generate_markdown_report(self, results: Dict[str, Dict], analysis_stats: Dict = None) -> None:
        """Final Markdown report."""
        report_path = self.output_dir / 'final_report.md'
        with open(report_path, 'w') as f:
            f.write("# YOLO Annotation QA Experiment Report\n\n")
            f.write("## 1. Experiment Results\n\n")
            f.write("| Experiment | mAP50 | mAP50-95 | Precision | Recall |\n")
            f.write("|---|---|---|---|---|\n")
            for exp, m in results.items():
                f.write(f"| {exp} | {self._fmt(m, 'mAP50')} | {self._fmt(m, 'mAP50-95')} | "
                        f"{self._fmt(m, 'precision')} | {self._fmt(m, 'recall')} |\n")
            seeds = {m.get('n_seeds') for m in results.values() if m.get('n_seeds')}
            if seeds:
                f.write(f"\nEach row is the mean ± standard deviation over {max(seeds)} training seeds.\n")

            f.write("\n## 2. Analysis\n\n")
            if all(k in results for k in ('Original', 'Noisy', 'Cleaned')):
                loss = results['Original']['mAP50'] - results['Noisy']['mAP50']
                recovery = results['Cleaned']['mAP50'] - results['Noisy']['mAP50']
                f.write(f"- mAP50 lost to noise (Original - Noisy): {loss:.4f}\n")
                f.write(f"- mAP50 recovered by cleaning (Cleaned - Noisy): {recovery:.4f}\n")
                if loss > 1e-9:
                    f.write(f"- Share of the loss recovered: {100 * recovery / loss:.1f}%\n")
                stds = [results[k].get('mAP50_std') for k in ('Original', 'Noisy', 'Cleaned')]
                if all(s is not None for s in stds):
                    noise_level = max(stds)
                    if abs(recovery) <= noise_level:
                        f.write("- The recovery is within one standard deviation of seed-to-seed variation, "
                                "so it should not be read as a real improvement.\n")
        logger.info(f"Markdown report saved to {report_path}")
