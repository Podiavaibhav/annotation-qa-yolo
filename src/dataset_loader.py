"""
Dataset loader for YOLO-format object detection datasets.
Supports standard YOLO directory structures and YAML configuration files.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


def _labels_for_images(images_path: Path) -> Path:
    """Swap the last 'images' folder in a path for 'labels' (the rule Ultralytics uses)."""
    parts = list(images_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == 'images':
            parts[i] = 'labels'
            return Path(*parts)
    return images_path / '__no_labels_folder__'   # sentinel that never exists


class YOLODatasetLoader:
    """
    Loads and validates YOLO-format object detection datasets.
    
    Expected structure:
    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize dataset loader.
        
        Args:
            dataset_path: Path to dataset directory or YAML file
        """
        self.dataset_path = Path(dataset_path)
        
        # If path is a YAML file, use its parent directory
        if self.dataset_path.suffix == '.yaml':
            self.yaml_path = self.dataset_path
            self.dataset_dir = self.dataset_path.parent
        else:
            self.dataset_dir = self.dataset_path
            self.yaml_path = self._find_yaml_file()
        
        self.config = self._load_config()
        self.classes = self.config.get('names', [])
        self.num_classes = self.config.get('nc', len(self.classes))
        
    def _find_yaml_file(self) -> Path:
        """Find YAML configuration file in dataset directory."""
        yaml_files = list(self.dataset_path.glob('*.yaml'))
        if not yaml_files:
            raise FileNotFoundError(
                f"No YAML configuration file found in {self.dataset_path}"
            )
        return yaml_files[0]
    
    def _load_config(self) -> Dict:
        """Load and validate YAML configuration."""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {self.yaml_path}")
        
        with open(self.yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        if 'names' not in config and 'nc' not in config:
            raise ValueError("YAML must contain 'names' or 'nc' field")
        
        return config
    
    _SPLIT_ALIASES = {'train': ['train'], 'val': ['val', 'valid', 'validation'], 'test': ['test']}

    def _dataset_root(self) -> Path:
        """Folder the yaml's relative split paths are relative to."""
        raw = self.config.get('path')
        if raw:
            p = Path(raw)
            if p.is_absolute():
                if p.exists():
                    return p
            elif (self.dataset_dir / p).exists():
                return self.dataset_dir / p
        return self.dataset_dir

    def get_split_paths(self, split: str = 'train') -> Tuple[Path, Path]:
        """
        Get paths to images and labels for a specific split.

        Handles the layouts Ultralytics datasets come in: images/<split>, <split>/images,
        and whatever the yaml points at (labels are found by swapping 'images' for 'labels').
        'val' also matches 'valid' and 'validation'.

        Args:
            split: Dataset split ('train', 'val', or 'test')

        Returns:
            Tuple of (images_path, labels_path)
        """
        root = self._dataset_root()
        names = self._SPLIT_ALIASES.get(split, [split])

        candidates = []
        for name in names:
            candidates.append((root / 'images' / name, root / 'labels' / name))
            candidates.append((root / name / 'images', root / name / 'labels'))
        for name in names:
            entry = self.config.get(name)
            if isinstance(entry, str):
                p = Path(entry)
                p = p if p.is_absolute() else root / p
                candidates.append((p, _labels_for_images(p)))   # yaml points at the images folder
                candidates.append((p / 'images', p / 'labels'))  # yaml points at the split folder

        for images_path, labels_path in candidates:
            if images_path.exists() and labels_path.exists():
                return images_path, labels_path

        raise FileNotFoundError(
            f"Could not find {split} split in {self.dataset_dir}"
        )

    def get_image_label_pairs(self, split: str = 'train') -> List[Tuple[Path, Path]]:
        """
        Get matched pairs of image and label files.
        
        Args:
            split: Dataset split
            
        Returns:
            List of (image_path, label_path) tuples
        """
        images_path, labels_path = self.get_split_paths(split)
        
        pairs = []
        for img_file in images_path.glob('*'):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            label_file = labels_path / (img_file.stem + '.txt')
            if label_file.exists():
                pairs.append((img_file, label_file))
            else:
                logger.warning(f"No label file for {img_file.name}")
        
        return pairs
    
    def load_annotations(self, label_path: Path) -> List[Dict]:
        """
        Load annotations from a YOLO label file.
        
        Args:
            label_path: Path to label TXT file
            
        Returns:
            List of annotation dictionaries with keys:
            - class_id: int
            - x_center: float (normalized)
            - y_center: float (normalized)
            - width: float (normalized)
            - height: float (normalized)
            - raw_line: str (original line for error reporting)
        """
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        annotations = []
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    logger.warning(
                        f"Invalid annotation format in {label_path.name} "
                        f"line {line_num}: expected 5 values, got {len(parts)}"
                    )
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    annotations.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'raw_line': line,
                        'line_number': line_num
                    })
                except ValueError as e:
                    logger.warning(
                        f"Non-numeric values in {label_path.name} "
                        f"line {line_num}: {e}"
                    )
        
        return annotations
    
    def get_dataset_stats(self) -> Dict:
        """
        Get basic dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_classes': self.num_classes,
            'class_names': self.classes,
            'splits': {}
        }
        
        for split in ['train', 'val', 'test']:
            try:
                pairs = self.get_image_label_pairs(split)
                stats['splits'][split] = {
                    'num_images': len(pairs),
                    'num_annotations': sum(
                        len(self.load_annotations(label_path))
                        for _, label_path in pairs
                    )
                }
            except FileNotFoundError:
                stats['splits'][split] = {
                    'num_images': 0,
                    'num_annotations': 0
                }
        
        return stats
    
    def validate_dataset(self) -> List[str]:
        """
        Validate dataset structure and integrity.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check YAML file
        if not self.yaml_path.exists():
            errors.append(f"YAML file not found: {self.yaml_path}")
        
        # Check splits
        for split in ['train', 'val']:
            try:
                images_path, labels_path = self.get_split_paths(split)
                
                if not images_path.exists():
                    errors.append(f"Images directory not found: {images_path}")
                
                if not labels_path.exists():
                    errors.append(f"Labels directory not found: {labels_path}")
                
                # Check for orphaned files
                if images_path.exists() and labels_path.exists():
                    image_stems = {f.stem for f in images_path.glob('*') 
                                   if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
                    label_stems = {f.stem for f in labels_path.glob('*.txt')}
                    
                    orphaned_images = image_stems - label_stems
                    orphaned_labels = label_stems - image_stems
                    
                    if orphaned_images:
                        errors.append(
                            f"{split}: {len(orphaned_images)} images without labels"
                        )
                    if orphaned_labels:
                        errors.append(
                            f"{split}: {len(orphaned_labels)} labels without images"
                        )
                        
            except FileNotFoundError as e:
                errors.append(str(e))
        
        return errors
