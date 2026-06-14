"""
analyzer.py

BDD100K Dataset Analyzer
"""

from __future__ import annotations
from parser import FrameRecord, BDDParser  
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Dict, Any, Tuple, Set, List

import numpy as np


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
LOGGER = logging.getLogger(__name__)

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT
MAX_GEOMETRY_SAMPLES = 5000
SMALL_OBJECT_THRESHOLD = 0.01
MEDIUM_OBJECT_THRESHOLD = 0.10


class BDDDatasetAnalyzer:
    """
    Analyzer for BDD100K dataset.
    """

    def __init__(self, split_name: str) -> None:
        self.split_name = split_name
        self.total_frames = 0
        self.total_objects = 0
        self.class_stats = defaultdict(
            lambda: {"count": 0, "images": set(), "areas": [], "aspect_ratios": []}
        )
        self.class_frequency = defaultdict(int)
        self.weather_distribution = defaultdict(int)
        self.scene_distribution = defaultdict(int)
        self.timeofday_distribution = defaultdict(int)
        self.box_areas = defaultdict(list)
        self.aspect_ratios = defaultdict(list)
        self.size_distribution = defaultdict(
            lambda: {"small": 0, "medium": 0, "large": 0}
        )
        self.occlusion_stats = defaultdict(int)
        self.truncation_stats = defaultdict(int)
        self.objects_per_frame = []
        self.geometry_samples = []
        self.co_occurrence = defaultdict(lambda: defaultdict(int))
        self.hard_samples = []
        self.interesting_samples = {
            "high_density_frames": [],
            "low_density_frames": [],
            "high_occlusion_frames": [],
            "high_truncation_frames": [],
            "largest_objects": [],
            "smallest_objects": [],
            "rarest_class_samples": defaultdict(list),
        }
        self.anomalies = {
            "micro_boxes": 0,
            "macro_boxes": 0,
            "extreme_aspects": 0,
            "empty_frames": 0,
        }

    @staticmethod
    def calculate_geometry(box: Any) -> Tuple[float, float, float, float]:
        """Calculate width, height, area, and aspect ratio of a bounding box.
        """
        width = max(box.x2 - box.x1, 1.0)
        height = max(box.y2 - box.y1, 1.0)
        return width, height, width * height, width / height

    @staticmethod
    def classify_size(area_ratio: float) -> str:
        """Classify object size based on area ratio."""
        if area_ratio < SMALL_OBJECT_THRESHOLD:
            return "small"
        if area_ratio < MEDIUM_OBJECT_THRESHOLD:
            return "medium"
        return "large"

    def analyze_dataset(self, frames: Iterable[FrameRecord]) -> None:
        """Main analysis loop over all frames."""
        LOGGER.info("Starting analysis of %s split", self.split_name)
        for frame in frames:
            self.total_frames += 1
            self._process_frame(frame)
        LOGGER.info(
            "Completed analysis. Frames=%d Objects=%d",
            self.total_frames,
            self.total_objects,
        )

    def _process_frame(self, frame: FrameRecord) -> None:
        """Process a single frame and update all statistics.""" 
        labels = frame.labels
        object_count = len(labels)
        self.total_objects += object_count
        self.objects_per_frame.append(object_count)

        if object_count == 0:
            self.anomalies["empty_frames"] += 1
        self._store_density_sample(frame.name, object_count, labels)

        categories = {label.category for label in labels if label.box2d is not None}
        self._update_co_occurrence(categories)

        attrs = frame.attributes
        if attrs.weather:
            self.weather_distribution[attrs.weather] += 1
        if attrs.scene:
            self.scene_distribution[attrs.scene] += 1
        if attrs.timeofday:
            self.timeofday_distribution[attrs.timeofday] += 1

        score = self._compute_hardness_score(frame, labels, object_count)
        self.hard_samples.append(
            {
                "name": frame.name,
                "score": score,
                "weather": attrs.weather,
                "timeofday": attrs.timeofday,
                "scene": attrs.scene,
                "objects": object_count,
            }
        )

        for label in labels:
            self._process_label(frame.name, label)

    def _process_label(self, frame_name: str, label: Any) -> None:
        """Process a single label and update statistics.""" 
        if label.box2d is None:
            return
        category = label.category
        _, _, area, aspect_ratio = self.calculate_geometry(label.box2d)
        area_ratio = area / IMAGE_AREA

        self.class_frequency[category] += 1
        if area_ratio < 0.001:
            self.anomalies["micro_boxes"] += 1
        if area_ratio > 0.80:
            self.anomalies["macro_boxes"] += 1
        if aspect_ratio > 5.0 or aspect_ratio < 0.2:
            self.anomalies["extreme_aspects"] += 1

        # self._track_object_extremes(frame_name, category, area_ratio, aspect_ratio)
        self._track_object_extremes(
            frame_name, category, area_ratio, aspect_ratio, label.box2d
        )

        stats = self.class_stats[category]
        stats["count"] += 1
        stats["images"].add(frame_name)
        stats["areas"].append(area)
        stats["aspect_ratios"].append(aspect_ratio)

        self.box_areas[category].append(area)
        self.aspect_ratios[category].append(aspect_ratio)

        size_bucket = self.classify_size(area_ratio)
        self.size_distribution[category][size_bucket] += 1

        if label.attributes:
            if getattr(label.attributes, "occluded", False):
                self.occlusion_stats[category] += 1
            if getattr(label.attributes, "truncated", False):
                self.truncation_stats[category] += 1

        self._store_geometry_sample(category, area_ratio, aspect_ratio, label.box2d)

    def _update_co_occurrence(self, categories: Set[str]) -> None:
        """Update co-occurrence counts for all pairs of categories in the same frame."""
        for cls1 in categories:
            for cls2 in categories:
                if cls1 != cls2:
                    self.co_occurrence[cls1][cls2] += 1

    def _store_density_sample(
        self, frame_name: str, object_count: int, labels: Any
    ) -> None:
        """Store samples of frames with high and low object density for visualization."""
        boxes = [
            [l.box2d.x1, l.box2d.y1, l.box2d.x2, l.box2d.y2] for l in labels if l.box2d
        ]
        categories = [l.category for l in labels if l.box2d]
        sample = {
            "name": frame_name,
            "objects": object_count,
            "boxes": boxes,
            "labels": categories,
        }
        self.interesting_samples["high_density_frames"].append(sample)
        self.interesting_samples["low_density_frames"].append(sample)

    """
    def _track_object_extremes(self, frame_name: str, category: str, area_ratio: float, aspect_ratio: float) -> None:
        sample = {"frame": frame_name, "category": category, "area_ratio": area_ratio}
        self.interesting_samples["largest_objects"].append(sample)
        self.interesting_samples["smallest_objects"].append(sample)
    """

    def _track_object_extremes(
        self,
        frame_name: str,
        category: str,
        area_ratio: float,
        aspect_ratio: float,
        box: Any,
    ) -> None:
        
        """
        Store object extremes together with bounding boxes
        so Streamlit can visualize them.
        """

        sample = {
            "name": frame_name,
            "category": category,
            "area_ratio": area_ratio,
            "aspect_ratio": aspect_ratio,
            "boxes": [[float(box.x1), float(box.y1), float(box.x2), float(box.y2)]],
            "labels": [category],
        }

        self.interesting_samples["largest_objects"].append(sample)
        self.interesting_samples["smallest_objects"].append(sample)

    def _compute_hardness_score(
        self, frame: FrameRecord, labels: Any, object_count: int
    ) -> float:
        """Compute a heuristic hardness score for a frame based on various factors."""
        score = 0.0
        weather = getattr(frame.attributes, "weather", None)
        timeofday = getattr(frame.attributes, "timeofday", None)

        if timeofday == "night":
            score += 3.0
        if weather in ["rainy", "foggy", "snowy"]:
            score += 3.0
        if object_count > 40:
            score += 2.0

        occluded, truncated, micro_boxes = 0, 0, 0
        for label in labels:
            if not label.box2d:
                continue
            if label.attributes:
                if getattr(label.attributes, "occluded", False):
                    occluded += 1
                if getattr(label.attributes, "truncated", False):
                    truncated += 1
            _, _, area, _ = self.calculate_geometry(label.box2d)
            if (area / IMAGE_AREA) < 0.001:
                micro_boxes += 1

        return round(
            score + (occluded * 0.2) + (truncated * 0.1) + (micro_boxes * 0.2), 2
        )

    def _store_geometry_sample(
        self, category: str, area_ratio: float, aspect_ratio: float, box: Any
    ) -> None:
        """Store random samples of object geometries for visualization."""
        center_x = (box.x1 + box.x2) / 2
        center_y = (box.y1 + box.y2) / 2
        sample = {
            "category": category,
            "area_ratio": area_ratio,
            "aspect_ratio": aspect_ratio,
            "center_x": center_x,
            "center_y": center_y,
        }
        if len(self.geometry_samples) < MAX_GEOMETRY_SAMPLES:
            self.geometry_samples.append(sample)
        else:
            idx = random.randint(0, len(self.geometry_samples) - 1)
            if random.random() < 0.05:
                self.geometry_samples[idx] = sample

    def finalize_analytics(self) -> None:
        """Finalize analytics by sorting and trimming interesting samples."""
        self.interesting_samples["largest_objects"] = sorted(
            self.interesting_samples["largest_objects"],
            key=lambda x: x["area_ratio"],
            reverse=True,
        )[:50]
        self.interesting_samples["smallest_objects"] = sorted(
            self.interesting_samples["smallest_objects"], key=lambda x: x["area_ratio"]
        )[:50]
        self.interesting_samples["high_density_frames"] = sorted(
            self.interesting_samples["high_density_frames"],
            key=lambda x: x["objects"],
            reverse=True,
        )[:50]
        self.interesting_samples["low_density_frames"] = sorted(
            self.interesting_samples["low_density_frames"], key=lambda x: x["objects"]
        )[:50]
        self.hard_samples = sorted(
            self.hard_samples, key=lambda x: x["score"], reverse=True
        )[:100]

    def build_summary(self) -> Dict[str, Any]:
        """Build a summary of the dataset statistics."""
        if not self.objects_per_frame:
            return {
                "total_frames": self.total_frames,
                "total_objects": self.total_objects,
                "avg_objects_per_frame": 0.0,
                "max_objects_per_frame": 0,
                "min_objects_per_frame": 0,
            }
        return {
            "total_frames": self.total_frames,
            "total_objects": self.total_objects,
            "avg_objects_per_frame": float(np.mean(self.objects_per_frame)),
            "max_objects_per_frame": int(np.max(self.objects_per_frame)),
            "min_objects_per_frame": int(np.min(self.objects_per_frame)),
        }

    def generate_adas_insights(self) -> List[str]:
        """Generate insights relevant for ADAS perception model development."""
        insights = []
        total_obj = max(self.total_objects, 1)
        sorted_classes = sorted(
            self.class_frequency.items(), key=lambda x: x[1], reverse=True
        )
        if sorted_classes:
            insights.append(
                f"{sorted_classes[0][0]} accounts for {(sorted_classes[0][1] / total_obj) * 100:.1f}% of all annotations."
            )
            insights.append(
                f"{sorted_classes[-1][0]} is the rarest class with only {(sorted_classes[-1][1] / total_obj) * 100:.3f}% of annotations."
            )
        night_pct = (
            self.timeofday_distribution.get("night", 0) / max(self.total_frames, 1)
        ) * 100
        insights.append(f"Night scenes represent {night_pct:.1f}% of all frames.")
        insights.append(
            f"{(self.anomalies['micro_boxes'] / total_obj) * 100:.2f}% of objects occupy less than 0.1% of the image."
        )
        if self.occlusion_stats:
            insights.append(
                f"{max(self.occlusion_stats.items(), key=lambda x: x[1])[0]} has the highest occlusion frequency."
            )
        return insights

    def export_class_statistics(self) -> Dict[str, Any]:
        """Export class-wise statistics including count, image frequency, mean area, and aspect ratio."""
        return {
            cls: {
                "count": s["count"],
                "images": len(s["images"]),
                "mean_area": float(np.mean(s["areas"])) if s["areas"] else 0.0,
                "mean_aspect_ratio": (
                    float(np.mean(s["aspect_ratios"])) if s["aspect_ratios"] else 0.0
                ),
            }
            for cls, s in self.class_stats.items()
        }

    def export_co_occurrence(self) -> Dict[str, Dict[str, int]]:
        """Export co-occurrence matrix for classes."""
        return {cls: dict(values) for cls, values in self.co_occurrence.items()}

    def export_metadata(self) -> Dict[str, Dict[str, int]]:
        """Export metadata distributions for weather, scene, and time of day."""
        return {
            "weather": dict(self.weather_distribution),
            "scene": dict(self.scene_distribution),
            "timeofday": dict(self.timeofday_distribution),
        }

    def export_visibility(self) -> Dict[str, Dict[str, int]]:
        """Export visibility statistics for occlusion and truncation."""
        return {
            "occlusion": dict(self.occlusion_stats),
            "truncation": dict(self.truncation_stats),
        }

    def export_size_distribution(self) -> Dict[str, Dict[str, int]]:
        """Export size distribution for each class."""
        return {cls: dict(values) for cls, values in self.size_distribution.items()}

    def export_interesting_samples(self) -> Dict[str, List[Any]]:
        """Export interesting samples for visualization."""
        return {
            "high_density_frames": self.interesting_samples["high_density_frames"],
            "low_density_frames": self.interesting_samples["low_density_frames"],
            "largest_objects": self.interesting_samples["largest_objects"],
            "smallest_objects": self.interesting_samples["smallest_objects"],
        }

    def export_report(self, output_file: str) -> None:
        """Export the complete analysis report to a JSON file."""
        LOGGER.info("Exporting report...")
        report = {
            "summary": self.build_summary(),
            "class_distribution": self.export_class_statistics(),
            "metadata_distribution": self.export_metadata(),
            "visibility_statistics": self.export_visibility(),
            "size_distribution": self.export_size_distribution(),
            "co_occurrence": self.export_co_occurrence(),
            "anomalies": self.anomalies,
            "hard_samples": self.hard_samples,
            "interesting_samples": self.export_interesting_samples(),
            "geometry_samples": self.geometry_samples,
            "adas_insights": self.generate_adas_insights(),
        }
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)
        LOGGER.info("Report saved to %s", output_file)


if __name__ == "__main__":
    CONFIG_PATH = r"E:\interview\BoschProject\data\bdd100k_labels_release\bdd100k\configs\bdd100k_2D_labels_config.json"

    # Analyze Train Split
    train_parser = BDDParser(
        annotation_file=r"E:\interview\BoschProject\data\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json",
        config_file=CONFIG_PATH,
    )
    train_analyzer = BDDDatasetAnalyzer(split_name="train")
    train_analyzer.analyze_dataset(train_parser.parse())
    train_analyzer.finalize_analytics()
    train_analyzer.export_report(
        r"E:\interview\BoschProject\data\bdd100k_labels_release\bdd100k\reports\train_report.json"
    )

    # Analyze Validation Split
    val_parser = BDDParser(
        annotation_file=r"E:\interview\BoschProject\data\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_val.json",
        config_file=CONFIG_PATH,
    )
    val_analyzer = BDDDatasetAnalyzer(split_name="val")
    val_analyzer.analyze_dataset(val_parser.parse())
    val_analyzer.finalize_analytics()
    val_analyzer.export_report(
        r"E:\interview\BoschProject\data\bdd100k_labels_release\bdd100k\reports\val_report.json"
    )
