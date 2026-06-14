import cv2
import random
from pathlib import Path

from parser import BDDParser


class AnnotationVisualizer:
    """
    Visualize BDD100K annotations using parser output.
    """

    def __init__(
        self,
        parser: BDDParser,
        image_dir: str,
    ) -> None:

        self.image_dir = Path(image_dir)

        self.frames = list(parser.parse())

        self.index = 0

        self.class_colors = self._generate_class_colors(
            parser.get_classes()
        )

    @staticmethod
    def _generate_class_colors(classes):
        """
        Generate consistent random colors for each class.
        """

        random.seed(42)

        colors = {}

        for cls in classes:
            colors[cls] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

        return colors

    def draw_frame(self, frame):

        image_path = self.image_dir / frame.name

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not load image: {image_path}")
            return None

        for label in frame.labels:

            if label.box2d is None:
                continue

            box = label.box2d

            color = self.class_colors.get(
                label.category,
                (255, 255, 255),
            )

            x1 = int(box.x1)
            y1 = int(box.y1)
            x2 = int(box.x2)
            y2 = int(box.y2)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                image,
                label.category,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        info = (
            f"Image {self.index + 1}/{len(self.frames)}"
            f" | Objects: {len(frame.labels)}"
        )

        cv2.putText(
            image,
            info,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        return image

    def run(self):

        cv2.namedWindow(
            "BDD100K Visualizer",
            cv2.WINDOW_NORMAL
        )

        while True:

            frame = self.frames[self.index]

            image = self.draw_frame(frame)

            if image is None:

                self.index += 1

                if self.index >= len(self.frames):
                    break

                continue

            cv2.imshow(
                "BDD100K Visualizer",
                image,
            )

            key = cv2.waitKey(0)

            # Right Arrow
            if key == 83 or key == ord("d"):

                self.index = min(
                    self.index + 1,
                    len(self.frames) - 1
                )

            # Left Arrow
            elif key == 81 or key == ord("a"):

                self.index = max(
                    self.index - 1,
                    0
                )

            # ESC
            elif key == 27:

                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = BDDParser(
        annotation_file=(
            r"data\bdd100k_labels_release\bdd100k\labels"
            r"\bdd100k_labels_images_train.json"
        ),
        config_file=(
            r"data\bdd100k_labels_release\bdd100k\configs"
            r"\bdd100k_2D_labels_config.json"
        ),
    )

    visualizer = AnnotationVisualizer(
        parser=parser,
        image_dir=(
            r".\data\bdd100k_images_100k\bdd100k\images\100k\train"
        ),
    )

    visualizer.run()