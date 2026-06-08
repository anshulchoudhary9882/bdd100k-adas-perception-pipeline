"""
parser.py

BDD100K Object Detection Parser
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Literal, Optional, List, Set

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

class Box2D(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    model_config = ConfigDict(extra="ignore")

class LabelAttributes(BaseModel):
    occluded: bool = False
    truncated: bool = False
    trafficLightColor: Optional[str] = None
    model_config = ConfigDict(extra="ignore")

class Label(BaseModel):
    id: str | int
    category: str
    box2d: Optional[Box2D] = None
    attributes: Optional[LabelAttributes] = None
    model_config = ConfigDict(extra="ignore")

class FrameAttributes(BaseModel):
    weather: Optional[Literal["clear", "overcast", "rainy", "snowy", "foggy", "partly cloudy", "undefined"]] = None
    scene: Optional[Literal["city street", "highway", "residential", "parking lot", "gas stations", "tunnel", "undefined"]] = None
    timeofday: Optional[Literal["daytime", "night", "dawn/dusk", "undefined"]] = None
    model_config = ConfigDict(extra="ignore")

class FrameRecord(BaseModel):
    name: str
    attributes: FrameAttributes
    timestamp: Optional[int] = None
    labels: List[Label] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")

class BDDParser:
    def __init__(self, annotation_file: str, config_file: str) -> None:
        self.annotation_file = Path(annotation_file)
        self.allowed_classes = self._load_allowed_classes(config_file)

    @staticmethod
    def _load_allowed_classes(config_file: str) -> Set[str]:
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        return set(config.get("allowed_detection_classes", []))

    def parse(self) -> Iterator[FrameRecord]:
        if not self.annotation_file.exists():
            raise FileNotFoundError(self.annotation_file)

        LOGGER.info("Parsing %s", self.annotation_file.name)
        with open(self.annotation_file, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        valid_frames = 0
        invalid_frames = 0

        for item in raw_data:
            try:
                frame = FrameRecord.model_validate(item)
                frame.labels = [
                    label for label in frame.labels
                    if (label.category in self.allowed_classes and label.box2d is not None)
                ]
                valid_frames += 1
                yield frame
            except ValidationError as exc:
                invalid_frames += 1
                LOGGER.warning("Validation failed for frame %s", item.get("name", "unknown"))
                LOGGER.debug(str(exc))

        LOGGER.info("Parsing complete. Valid=%d Invalid=%d", valid_frames, invalid_frames)

    def count_frames(self) -> int:
        with open(self.annotation_file, "r", encoding="utf-8") as file:
            return len(json.load(file))

    def get_classes(self) -> List[str]:
        return sorted(self.allowed_classes)

if __name__ == "__main__":
    parser = BDDParser(
        annotation_file="data\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json",
        config_file="data\bdd100k_labels_release\bdd100k\configs\bdd100k_2D_labels_config.json"
    )

    LOGGER.info("Allowed classes:")
    for cls in parser.get_classes():
        LOGGER.info(" - %s", cls)

    for idx, frame_record in enumerate(parser.parse()):
        LOGGER.info("%s | Objects=%d", frame_record.name, len(frame_record.labels))
        if idx == 5: break