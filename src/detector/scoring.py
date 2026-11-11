"""
Score the detector against the noise that was injected.

We know exactly which labels were corrupted (corruption_manifest.json), so we can measure
  recall     - of the errors we injected, how many did the detector flag?
  precision  - of the detector's flags, how many point at an injected error?

Precision is measured against injected errors only. A flag on a mistake that was already in the
original dataset counts as a false positive here, so precision is a lower bound.
"""
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SAME_BOX = 0.9   # two boxes this similar are "the same box"


def _iou(a: List[float], b: List[float]) -> float:
    """IoU of two (cx, cy, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _box(d: Dict) -> List[float]:
    return [d['x'], d['y'], d['w'], d['h']]


def _group(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[Path(str(r['image'])).stem].append(r)
    return grouped


def _read(path: Path) -> List[Dict]:
    return pd.read_csv(path).to_dict('records') if path.exists() else []


def score_detector(manifest_path: str, audit_dir: str, analysis_report_path: Optional[str] = None,
                   output_dir: Optional[str] = None, iou_thresh: float = 0.5,
                   loc_iou: float = 0.7) -> Dict:
    manifest = json.loads(Path(manifest_path).read_text())
    audit = Path(audit_dir)
    missing = _group(_read(audit / 'missing_annotations.csv'))
    classes = _group(_read(audit / 'class_disagreements.csv'))
    locs = _group(_read(audit / 'localization_disagreements.csv'))

    dups, have_analysis = {}, False
    if analysis_report_path and Path(analysis_report_path).exists():
        have_analysis = True
        report = json.loads(Path(analysis_report_path).read_text())
        dups = _group([s for s in report.get('suspicious_details', []) if s.get('issue') == 'exact_duplicate'])

    def pred_box(r):
        return [r['pred_cx'], r['pred_cy'], r['pred_w'], r['pred_h']]

    def gt_box(r):
        return [r['gt_cx'], r['gt_cy'], r['gt_w'], r['gt_h']]

    # ------------------------------------------------------------ recall
    recall = {}

    def tally(name, injected, caught, extra=None):
        recall[name] = {'injected': injected, 'caught': caught,
                        'recall': (caught / injected) if injected else None, **(extra or {})}

    deleted = [e for e in manifest if e['type'] == 'deleted']
    caught = correct = 0
    for e in deleted:
        stem = Path(e['file']).stem
        hits = [r for r in missing.get(stem, []) if _iou(pred_box(r), _box(e['original'])) >= iou_thresh]
        caught += bool(hits)
        correct += any(int(r['pred_class']) == e['original']['cls'] for r in hits)
    tally('deleted', len(deleted), caught, {'restored_with_correct_class': correct})

    flips = [e for e in manifest if e['type'] == 'class_flip']
    caught = correct = 0
    for e in flips:
        stem = Path(e['file']).stem
        hits = [r for r in classes.get(stem, [])
                if _iou(gt_box(r), _box(e['result'])) >= SAME_BOX and int(r['gt_class']) == e['result']['cls']]
        caught += bool(hits)
        correct += any(int(r['pred_class']) == e['original']['cls'] for r in hits)
    tally('class_flip', len(flips), caught, {'diagnosed_with_correct_class': correct})

    geometric = [e for e in manifest if e['type'] in ('coordinate_shift', 'resize')]
    real, mild = [], 0
    for e in geometric:
        iou_orig = e.get('iou_with_original')
        if iou_orig is None:
            iou_orig = _iou(_box(e['original']), _box(e['result']))
        if iou_orig < loc_iou:
            real.append(e)
        else:
            mild += 1
    caught = fixed = 0
    for e in real:
        stem = Path(e['file']).stem
        hits = [r for r in locs.get(stem, []) if _iou(gt_box(r), _box(e['result'])) >= SAME_BOX]
        caught += bool(hits)
        fixed += any(_iou(pred_box(r), _box(e['original'])) >= iou_thresh for r in hits)
    tally('shift_or_resize', len(real), caught,
          {'predicted_box_matches_original': fixed, 'excluded_mild': mild})

    duplicates = [e for e in manifest if e['type'] == 'duplicate']
    if have_analysis:
        caught = sum(
            any(_iou(list(r['box']), _box(e['result'])) >= SAME_BOX for r in dups.get(Path(e['file']).stem, []))
            for e in duplicates)
        tally('duplicate', len(duplicates), caught)
    else:
        recall['duplicate'] = {'injected': len(duplicates), 'caught': None, 'recall': None}

    # ------------------------------------------------------------ precision
    events_by = defaultdict(list)
    for e in manifest:
        events_by[(e['type'], Path(e['file']).stem)].append(e)

    precision = {}
    precision['missing'] = _precision(
        missing, lambda stem: events_by.get(('deleted', stem), []),
        lambda r, e: _iou(pred_box(r), _box(e['original'])) >= iou_thresh)
    precision['class_disagreement'] = _precision(
        classes, lambda stem: events_by.get(('class_flip', stem), []),
        lambda r, e: _iou(gt_box(r), _box(e['result'])) >= SAME_BOX and int(r['gt_class']) == e['result']['cls'])
    precision['localization'] = _precision(
        locs, lambda stem: events_by.get(('coordinate_shift', stem), []) + events_by.get(('resize', stem), []),
        lambda r, e: _iou(gt_box(r), _box(e['result'])) >= SAME_BOX)

    result = {'recall': recall, 'precision': precision}
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'detector_eval.json').write_text(json.dumps(result, indent=2))
        (out / 'detector_eval.md').write_text(_markdown(result))
    return result


def _precision(flags_by_stem, events_for_stem, matches) -> Dict:
    total = ok = 0
    for stem, rows in flags_by_stem.items():
        events = events_for_stem(stem)
        for r in rows:
            total += 1
            ok += any(matches(r, e) for e in events)
    return {'flags': total, 'matched_injected_error': ok, 'precision': (ok / total) if total else None}


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _markdown(result: Dict) -> str:
    lines = ["# Detector evaluation (against injected noise)", "",
             "Recall: share of injected errors that the detector flagged.", "",
             "| Injected error | Injected | Flagged | Recall | Extra |", "|---|---:|---:|---:|---|"]
    for name, r in result['recall'].items():
        extras = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in r.items()
                           if k not in ('injected', 'caught', 'recall'))
        lines.append(f"| {name} | {r['injected']} | {r['caught'] if r['caught'] is not None else 'n/a'} "
                     f"| {_pct(r['recall'])} | {extras} |")
    lines += ["", "Precision: share of the detector's flags that point at an injected error.", "",
              "| Flag type | Flags | Matched an injected error | Precision |", "|---|---:|---:|---:|"]
    for name, p in result['precision'].items():
        lines.append(f"| {name} | {p['flags']} | {p['matched_injected_error']} | {_pct(p['precision'])} |")
    lines += ["", "Precision counts flags on mistakes that were already in the original dataset as wrong, "
              "so it is a lower bound."]
    return "\n".join(lines) + "\n"
