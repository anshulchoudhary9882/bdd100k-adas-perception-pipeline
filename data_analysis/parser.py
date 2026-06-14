"""
parser.py

BDD100K Object Detection Parser
"""

#from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Literal, Optional, List, Set

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
LOGGER = logging.getLogger(__name__)


# represents a 2D bounding box in the BDD100K dataset
class Box2D(BaseModel):
    """
    Represents a 2D bounding box in image coordinates.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    model_config = ConfigDict(
        extra="ignore"
    )  # prevent errors from missing fields in some labels


# represents the attributes of a single object label in the BDD100K dataset
class LabelAttributes(BaseModel):
    """
    Stores additional object attributes such as
    occlusion, truncation, and traffic light color.
    """
    occluded: bool = False
    truncated: bool = False
    trafficLightColor: Optional[str] = (
        None  # only for traffic light category, can be "red", "yellow", "green", or "off"
    )
    model_config = ConfigDict(extra="ignore")


# represents a single object label in the BDD100K dataset
class Label(BaseModel):
    """Represents a single object label with category, bounding box, and attributes.
    """
    id: str | int
    category: str
    box2d: Optional[Box2D] = (
        None  # some labels may not have box2d if they are not annotated with bounding boxes
    )
    attributes: Optional[LabelAttributes] = (
        None  # some labels may not have attributes if they are not annotated with additional information
    )
    model_config = ConfigDict(extra="ignore")


# represents the attributes of a single frame in the BDD100K dataset
class FrameAttributes(BaseModel):
    """
    Stores environmental metadata such as weather,
    scene type, and time of day.
    """
    weather: Optional[
        Literal[
            "clear", "overcast", "rainy", "snowy", "foggy", "partly cloudy", "undefined"
        ]
    ] = None
    scene: Optional[
        Literal[
            "city street",
            "highway",
            "residential",
            "parking lot",
            "gas stations",
            "tunnel",
            "undefined",
        ]
    ] = None
    timeofday: Optional[Literal["daytime", "night", "dawn/dusk", "undefined"]] = None
    model_config = ConfigDict(extra="ignore")


#  represents a single frame in the BDD100K dataset
class FrameRecord(BaseModel):
    """
    Represents a single image frame and its annotations.
    """
    name: str
    attributes: FrameAttributes
    timestamp: Optional[int] = None
    labels: List[Label] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class BDDParser:
    """
    Parser for BDD100K object detection annotations.
    Provides validation and class filtering.
    """
    def __init__(self, annotation_file: str, config_file: str) -> None:
        self.annotation_file = Path(annotation_file)
        self.allowed_classes = self._load_allowed_classes(config_file)

    @staticmethod  # loads the allowed classes from a JSON config file
    def _load_allowed_classes(config_file: str) -> Set[str]:
        """
        Load allowed object classes from configuration file.

        Args:
            config_file: Path to JSON configuration.

        Returns:
            Set of allowed detection classes.
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        return set(config.get("allowed_detection_classes", []))

    def parse(self) -> Iterator[FrameRecord]:
        """
        Parse annotation file and yield validated
        FrameRecord objects one at a time.

        Yields:
            FrameRecord
        """
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
                    label
                    for label in frame.labels
                    if (
                        label.category in self.allowed_classes
                        and label.box2d is not None
                    )
                ]
                valid_frames += 1
                yield frame  #   yield the validated frame
            except ValidationError as exc:
                invalid_frames += 1
                LOGGER.warning(
                    "Validation failed for frame %s", item.get("name", "unknown")
                )
                LOGGER.debug(str(exc))

        LOGGER.info(
            "Parsing complete. Valid=%d Invalid=%d", valid_frames, invalid_frames
        )

    def count_frames(self) -> int:
        """
        Return total number of frames in the annotation file.
        """
        with open(self.annotation_file, "r", encoding="utf-8") as file:
            return len(json.load(file))

    def get_classes(self) -> List[str]:
        """
        Return sorted list of allowed classes.
        """
        return sorted(self.allowed_classes)


if __name__ == "__main__":
    parser = BDDParser(
        annotation_file="./data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json",
        config_file="./data/bdd100k_labels_release/bdd100k/configs/bdd100k_2D_labels_config.json",
    )

    LOGGER.info("Allowed classes:")
    for cls in parser.get_classes():
        LOGGER.info(" - %s", cls)

    for idx, frame_record in enumerate(parser.parse()):
        LOGGER.info("%s | Objects=%d", frame_record.name, len(frame_record.labels))
        if idx == 5:
            break
